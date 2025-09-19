import torch
from torchvision.utils import make_grid, save_image
import json
from model_architecture import UNetCond, UnetCondTime
from train import get_schedule, extract
import argparse
from tqdm import tqdm


def gen_denoise_process(args):
    device = "cuda"
    label_set = ["red sphere", "cyan cylinder", "cyan cube"]

    with open("objects.json", "r") as f:
        obj2idx = json.load(f)
        num_classes = len(obj2idx)


    use_improv_unet = args.use_improv_unet
    model_path = args.model_path
    T = args.T


    label_vec =  torch.zeros(num_classes)

    for label in label_set:
        label_vec[obj2idx[label]] = 1

    cond = label_vec.unsqueeze(0).to(device)


    if use_improv_unet:
        model = UnetCondTime(args.GN).to(device)
    else:
        model = UNetCond().to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    betas, alphas, alpha_cumprod = get_schedule(T)
    betas, alphas, alpha_cumprod = [x.to(device) for x in [betas, alphas, alpha_cumprod]]

    B = cond.shape[0]
    x_t = torch.randn(B, 3, 64, 64).to(device)

    denoise_process_imgs = []
    snapshot_ts = [31, 62, 93, 124, 155, 186, 217, 248, 279]


    for t in tqdm(reversed(range(T))):
        if t in snapshot_ts:
            denoise_process_imgs.append(x_t[0].clone().cpu())

        t_tensor = torch.full((B,), t, dtype=torch.long).to(device)
        
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


    denoise_process_imgs.append(x_t[0].clone().cpu())

    denoise_process_imgs = torch.stack(denoise_process_imgs)
    # grid = make_grid(denoise_process_imgs, nrow=n_img, normalize=True)
    grid = make_grid(denoise_process_imgs, nrow=10, normalize=False)
    save_image(grid, fp=args.output_name)


if __name__ == "__main__":
    # snapshot_ts = [ i * (T // n_img) for i in range(1, n_img-1)] + [T-1]
    # print(snapshot_ts)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="result_ep500_improvUnet_GN\checkpoint_ep500.pth")
    parser.add_argument("--T", type=int, default=1000)  
    parser.add_argument("--use-improv-unet", action="store_true", default=False)   
    parser.add_argument("--GN", action="store_true", default=False)
    parser.add_argument("--output-name",  type=str , default="denoising_process.png")   
    args = parser.parse_args()
    gen_denoise_process(args=args)
