import torch
import numpy as np
import gymnasium as gym
from dqn import DQN, AtariPreprocessor  # 確保你 `dqn.py` 有定義這些 class
import argparse

def evaluate_model(model_path, num_episodes=20, render=False):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    num_actions = env.action_space.n
    preprocessor = AtariPreprocessor()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 建立模型並載入訓練參數
    model = DQN(num_actions).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    rewards = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        state = preprocessor.reset(obs)
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = preprocessor.step(next_obs)

        print(f"Episode {ep + 1}: Reward = {total_reward}")
        rewards.append(total_reward)

    env.close()
    avg_reward = np.mean(rewards)
    print(f"\nAverage reward over {num_episodes} episodes: {avg_reward:.2f}")
    return avg_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained .pt model file")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--render", action="store_true", help="Render environment while evaluating")
    args = parser.parse_args()

    evaluate_model(args.model_path, args.num_episodes, args.render)
