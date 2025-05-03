import torch
import os
import json
from torchvision.utils import save_image
from model_architecture import UNetCond, UnetCondTime
from train import get_schedule, extract
from tqdm import tqdm
import argparse


@torch.no_grad()
def sample_images(model, cond, T, betas, alphas, alpha_cumprod, device):
    model.eval()
    B = cond.shape[0]
    x_t = torch.randn(B, 3, 64, 64).to(device)

    for t in reversed(range(T)):
        t_tensor = torch.full((B,), t, dtype=torch.long).to(device)
        # print(t_tensor.shape)
        eps_pred = model(x_t, t_tensor, cond)

        alpha = extract(alphas, t_tensor, x_t.shape)
        alpha_bar = extract(alpha_cumprod, t_tensor, x_t.shape)
        beta = extract(betas, t_tensor, x_t.shape)

        mean = (1 / alpha.sqrt()) * (x_t - ((1 - alpha) / (1 - alpha_bar).sqrt()) * eps_pred)

        if t > 0:
            noise = torch.randn_like(x_t)
            x_t = mean + beta.sqrt() * noise
        else:
            x_t = mean
    return x_t



def load_conditions(json_path, objects_path="objects.json"):
    with open(objects_path, "r") as f:
        obj2idx = json.load(f)
    num_classes = len(obj2idx)

    with open(json_path, "r") as f:
        raw_data = json.load(f)

    conds = []
    for cond in raw_data:
        vec = torch.zeros(num_classes)
        for label in cond:
            vec[obj2idx[label]] = 1.0
        conds.append(vec)
    return torch.stack(conds)  # (N, 24)

@torch.no_grad()
def generate_images(cond_tensor, save_dir, model, T, betas, alphas, alpha_cumprod, device):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    for idx in tqdm(range(len(cond_tensor)), desc=f"Generating to {save_dir}"):
        cond = cond_tensor[idx].unsqueeze(0).to(device)  # shape: (1, 24)
        img = sample_images(model, cond, T, betas, alphas, alpha_cumprod, device)  # (1, 3, 64, 64)
        save_image(img[0], os.path.join(save_dir, f"{idx}.png"))

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = args.T
    
    if args.use_improv_unet:
        model = UnetCondTime(group_normalize=args.GN).to(device)
    else:
        model = UNetCond().to(device)

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    betas, alphas, alpha_cumprod = get_schedule(T)
    betas, alphas, alpha_cumprod = [x.to(device) for x in [betas, alphas, alpha_cumprod]]

    for split in ["test", "new_test"]:
        conds = load_conditions(f"{split}.json")
        generate_images(conds, save_dir=f"{args.output_dir}/{split}", model=model, T=T,
                        betas=betas, alphas=alphas, alpha_cumprod=alpha_cumprod, device=device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)  
    parser.add_argument("--output-dir", type=str, default="./images")  
    parser.add_argument("--T", type=int, default=1000)  
    parser.add_argument("--use-improv-unet", action="store_true", default=False)
    parser.add_argument("--GN", action="store_true", default=False)


    args =  parser.parse_args()
    main(args=args)
