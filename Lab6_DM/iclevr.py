import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json
import os
from PIL import Image

class IclevrDataset(Dataset):
    def __init__(self, objects_fp="./objects.json", train_data_fp="./train.json", images_dir="./iclevr"):
        super().__init__()
        with open(objects_fp, "r") as obj_f:
            self.obj_codes = json.load(obj_f)
            self.n_code = len(self.obj_codes)
        with open(train_data_fp, "r") as data_f:
            self.data = list(json.load(data_f).items())
        self.images_dir = images_dir
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),         
            transforms.ToTensor(),               
        ])

    def __getitem__(self, index):
        sample = self.data[index]
        image_path = os.path.join(self.images_dir, sample[0])
        image = Image.open(image_path).convert("RGB")
        # image.show()
        image = self.transform(image)

        sample_labels = sample[1]
        label_vec = torch.zeros(self.n_code)
        for label in sample_labels:
            label_vec[self.obj_codes[label]] = 1.0

        return image, label_vec

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    # with open("./train.json", "r") as data_f:
    #     data = list(json.load(data_f).items())
    # print(data[0])

    dataset = IclevrDataset()
    print(len(dataset))
    sample = dataset[0]
    print(type(sample[0]))
    print(type(sample[1]))
    print(len(sample[1]))

