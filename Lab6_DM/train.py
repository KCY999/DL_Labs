import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


import argparse
import os

from model_architecture import UNetCond, UnetCondTime
import iclevr




def get_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, T)
    alphas = 1. - betas
    alpha_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_cumprod

def extract(a, t, x_shape):
    return a.gather(-1, t).view(-1, 1, 1, 1).expand(x_shape)

def q_sample(x0, t, noise, alpha_cumprod):
    sqrt_alpha = extract(alpha_cumprod.sqrt(), t, x0.shape)
    sqrt_one_minus = extract((1 - alpha_cumprod).sqrt(), t, x0.shape)
    return sqrt_alpha * x0 + sqrt_one_minus * noise



def train(args):
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    T = args.T
    saved_dir = args.saved_dir
    os.makedirs(saved_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.use_improv_unet:
        model = UnetCondTime(group_normalize=args.GN).to(device)
    else:
        model = UNetCond().to(device)


    optimizer = optim.Adam(model.parameters(), lr=lr)
    if args.use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    loss_fn = nn.MSELoss()

    # noise schedule
    betas, alphas, alpha_cumprod = get_schedule(T)
    betas, alphas, alpha_cumprod = [x.to(device) for x in [betas, alphas, alpha_cumprod]]


    dataset = iclevr.IclevrDataset()  
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    for epoch in tqdm(range(epochs)):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch}")

        for x0, cond in pbar:
            x0 = x0.to(device)
            cond = cond.to(device)

            # sample t, noise
            t = torch.randint(0, T, (x0.size(0),), device=device)
            noise = torch.randn_like(x0)

            # forward process
            x_t = q_sample(x0, t, noise, alpha_cumprod)


            pred_noise = model(x_t, t, cond)

            # Loss
            loss = loss_fn(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            pbar.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])

        if args.use_lr_scheduler:
            scheduler.step() 

        if (epoch + 1) % 100 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"[Epoch {epoch + 1}] current LR = {current_lr} \n")
            torch.save(model.state_dict(), os.path.join(saved_dir, f"checkpoint_ep{epoch+1}.pth"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved-dir", type=str, default="./results")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--use-lr-scheduler", action="store_true", default=False)
    parser.add_argument("--use-improv-unet", action="store_true", default=False)
    parser.add_argument("--GN", action="store_true", default=False)

    args = parser.parse_args()

    train(args=args)