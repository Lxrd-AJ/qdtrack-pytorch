import torch
import os
import random
import torchvision.transforms as transforms
from dataset.mot_dataset import MOTDataset

torch.manual_seed(100)

image_transforms = transforms.Compose([
    transforms.Resize((360, 720)), #(h,w)
    # transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

cur_dir = os.path.dirname(os.path.realpath(__file__))
mot_dataset_dir = os.path.join(os.path.dirname(cur_dir), "MOT17", "train")
train_dataset = MOTDataset(mot_dataset_dir, image_tf=image_transforms, item_tf = None)

if __name__ == "__main__":
    idx = random.randint(0, len(train_dataset))
    (im1, im2) = train_dataset[idx]
    im1.show()
    im2.show()