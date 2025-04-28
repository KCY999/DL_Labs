# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

gym.register_envs(ale_py)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        return self.network(x / 255.0)


class AtariPreprocessor:
    """ 
        Preprocesing the state input of DQN for Atari
    """   
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
        self.is_vector_input = False

    def preprocess(self, obs):
        if len(obs.shape) == 1: 
            self.is_vector_input = True
            return obs

        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        if self.is_vector_input:
            return frame  
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        if self.is_vector_input:
            return frame
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        # self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def __len__(self): 
        return len(self.buffer)

    def add(self, transition, error):
        ########## YOUR CODE HERE (for Task 3) ########## 
        priority = (abs(error) + 1e-7) ** self.alpha
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity       
        ########## END OF YOUR CODE (for Task 3) ########## 

    def sample(self, batch_size, beta):
        ########## YOUR CODE HERE (for Task 3) ########## 
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
        

        priorities = np.maximum(priorities, 1e-7)
        
        sampling_probs = priorities / np.sum(priorities)


        indices = np.random.choice(len(self.buffer), batch_size, p=sampling_probs, replace=True)
        
        samples = [self.buffer[i] for i in indices]

        weights = (len(self.buffer) * sampling_probs [indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, torch.tensor(weights, dtype=torch.float32)      
        ########## END OF YOUR CODE (for Task 3) ########## 

    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ########## 
        errors = errors.detach().cpu().numpy()
        for i, e in zip(indices, errors):
            self.priorities[i] = (abs(e) + 1e-7) ** self.alpha
        ########## END OF YOUR CODE (for Task 3) ########## 
        

class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)



        self.q_net =  DQN(4, self.num_actions).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net =  DQN(4, self.num_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr, eps=args.adam_eps)
        self.scheduler = StepLR(self.optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma) 

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21  # Initilized to 0 for CartPole and to -21 for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)


        # self.memory = deque(maxlen=args.memory_size) # replay buffer
        self.memory = PrioritizedReplayBuffer(capacity=args.memory_size)

        self.n_step = args.n_step
        self.n_step_buffer = deque(maxlen=self.n_step)


        self.beta_start = 0.4
        self.beta_end = 1.0
        self.beta_anneal_steps = 1000000

        self.use_reward_shape = args.use_reward_shape


    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=1000):
        for ep in range(episodes):
            obs, _ = self.env.reset()

            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                if self.use_reward_shape:
                    if reward == 1:
                        reward = 1.5
                    elif reward == -1:
                        reward = -2.0

                next_state = self.preprocessor.step(next_obs)
                self.n_step_buffer.append((state, action, reward, next_state, done))
                if len(self.n_step_buffer) == self.n_step:
                    n_state, n_action, n_reward, n_next_state, n_done = self._get_n_step_info()
                    max_prio = self.memory.priorities.max() if len(self.memory) > 0 else 1.0
                    self.memory.add((n_state, n_action, n_reward, n_next_state, n_done), error=max_prio)

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 1000 == 0:       
                    current_lr = self.optimizer.param_groups[0]['lr']
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f} LR: {current_lr:.6f}")
                    
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon,
                        "LR": current_lr   
                    })
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed 
                    
                    ########## END OF YOUR CODE ##########  
                if self.env_count in [200000, 400000, 600000, 800000, 1000000]:
                    model_path = os.path.join(self.save_dir, f"model_{self.env_count//1000}k.pt")
                    torch.save(self.q_net.state_dict(), model_path)
            
                    print(f"Saved snapshot at {self.env_count} steps to {model_path}")

            while len(self.n_step_buffer) > 0:
                n_state, n_action, n_reward, n_next_state, n_done = self._get_n_step_info()
                max_prio = self.memory.priorities.max() if len(self.memory) > 0 else 1.0
                self.memory.add((n_state, n_action, n_reward, n_next_state, n_done), error=max_prio)
                self.n_step_buffer.popleft()

            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            
            ########## END OF YOUR CODE ##########  
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })



    def evaluate(self, num_episodes=5):
        rewards = []
        for _ in range(num_episodes):
            obs, _ = self.test_env.reset()
            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.q_net(state_tensor).argmax().item()
                next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = self.preprocessor.step(next_obs)
            
            rewards.append(total_reward)
       
        return np.mean(rewards)

        




    def train(self):

        if len(self.memory) < self.replay_start_size:
            return 
        
        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1
       
        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer
        fraction = min(1.0, self.train_count / self.beta_anneal_steps)
        beta = self.beta_start + fraction * (self.beta_end - self.beta_start)

        samples, indices, weights = self.memory.sample(self.batch_size, beta)
        states, actions, rewards, next_states, dones = zip(*samples)
        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates 
        with torch.no_grad():
            next_q_values = self.q_net(next_states)
            next_actions = next_q_values.argmax(1).unsqueeze(1)
            
            next_q_values_target = self.target_net(next_states)
            next_q_target = next_q_values_target.gather(1, next_actions).squeeze(1)
            # print(rewards.shape, dones.shape, next_q_target.shape)
            target_q = rewards + (self.gamma ** self.n_step) * (1 - dones) * next_q_target
            # print(next_q_target.shape, target_q.shape)

        td_errors = torch.abs(target_q.detach() - q_values.detach())
        self.memory.update_priorities(indices, td_errors)

        # loss_per_sample = (q_values - target_q).pow(2)
        # loss = (loss_per_sample * weights.to(self.device).detach()).mean()

        # print(q_values.shape, target_q.shape) # torch.Size([32]) torch.Size([32])
        

        weights = weights.to(self.device)
        sample_losses = F.mse_loss(q_values, target_q, reduction='none')
        # print(weights.shape, sample_losses.shape)
        loss = (weights * sample_losses).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()


        ########## END OF YOUR CODE ##########  
        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        if self.train_count % 1000 == 0:
           print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f} ")
           wandb.log({
                "Train/Loss": loss.item(),
                "Train/Epsilon": self.epsilon,
                "Train/Beta": beta,
                "Train/Q_mean": q_values.mean().item(),
                "Train/Q_std": q_values.std().item(),
            })

    def _get_n_step_info(self):
        R = 0.0
        next_state, done = self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]

        for idx, (_, _, r, s_next, d) in enumerate(self.n_step_buffer):
            R += (self.gamma ** idx) * r
            if d: 
                next_state = s_next
                done = True
                break

        state, action = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
        return state, action, R, next_state, done

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, default="./pong_results_task3_v1")
    parser.add_argument("--wandb-run-name", type=str, default="Pong-full-run")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999999)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument("--n-step", type=int, default=5)
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta", type=float, default=0.4)
    parser.add_argument("--adam-eps", type=float, default=1e-8)
    parser.add_argument("--scheduler-step-size", type=float, default=100000)
    parser.add_argument("--scheduler-gamma", type=float, default=1)
    parser.add_argument("--use-reward-shape", action="store_true", default=False)
    args = parser.parse_args()

    wandb.init(project="DLP-Lab5-DQN-Pong-task3", name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(env_name="ALE/Pong-v5", args=args)
    # agent.run(episodes=1200)
    agent.run()