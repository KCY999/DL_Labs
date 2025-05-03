import os
import torch
from torchvision import transforms
from PIL import Image
from evaluator import evaluation_model
from iclevr import IclevrDataset
from torch.utils.data import Dataset, DataLoader
import json
import argparse
from tqdm import tqdm
import numpy as np

class EvalDataset(Dataset):
    def __init__(self, test_name="test", objects_fp="./objects.json", images_dir="./images"):
        super().__init__()
        test_label_fp = f"{test_name}.json"
        self.images_dir = os.path.join(images_dir, test_name)

        with open(objects_fp, "r") as obj_f:
            self.obj_codes = json.load(obj_f)
            self.n_code = len(self.obj_codes)

        with open(test_label_fp, "r") as data_f:
            self.data = [ (f"{i}.png", labels) for i, labels in enumerate(list(json.load(data_f)))]
            # print(len(self.data))

        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),         
            transforms.ToTensor(),   
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        sample = self.data[index]

        image_name = sample[0]
        image_path = os.path.join(self.images_dir, image_name)
        image = Image.open(image_path).convert("RGB")
        # image.show()

        image = self.transform(image)


        sample_labels = sample[1]

        label_vec = torch.zeros(self.n_code)
        for label in sample_labels:
            label_vec[self.obj_codes[label]] = 1.0

        return image, label_vec

    def __len__(self):
        # print(f"[DEBUG] Dataset length: {len(self.data)}")
        return len(self.data)



def evaluate(args):
    device = args.device
    evaluator = evaluation_model()

    eval_dataset = EvalDataset(test_name=args.test_name, images_dir=args.images_dir)
    eval_loader = DataLoader(dataset=eval_dataset, batch_size=4)

    all_acc = []
    i = 1
    for images, labels in eval_loader: 
        images = images.to(device)
        labels = labels.to(device)
        acc = evaluator.eval(images, labels)
        all_acc.append(acc)
        print(f"[BATCH {i}] batch score = {acc:.5}")
        i += 1

    avg_acc = np.mean(all_acc)
    print(f"[TOTAL] avg score = {avg_acc:.5}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-name", type=str, default="test")   
    parser.add_argument("--images-dir", type=str, default="images")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    evaluate(args=args)


