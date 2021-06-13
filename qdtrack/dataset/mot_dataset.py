import os
import itertools
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

class MOTDataset(Dataset):
    def __init__(self, dir, image_tf, item_tf) -> None:
        super().__init__()
        self.root_dir = dir
        self.seq_dirs = unique_sequences_from(dir)
        self.num_sequences = len(self.seq_dirs)
        self.image_transform = image_tf
        self.item_transform = item_tf

    def __len__(self) -> int:
        len_images = [len(images_in_sequence(seq)) for seq in self.seq_dirs]
        return sum(len_images)

    def __getitem__(self, idx):
        (seq_num, img_idx) = self.indexer(idx)
        img1_url = self.seq_indexer(seq_num, img_idx)
        img1 = Image.open(img1_url)

        interval = random.choice([-3,-2,-1,1,2,3])
        seq_maximum = [len(images_in_sequence(seq)) for seq in self.seq_dirs if seq_num in seq][0]
        next_img_idx = img_idx + interval
        if next_img_idx < 0 or next_img_idx >= seq_maximum:
            next_img_idx = img_idx + (-1 * interval) #flip in other direction
        img2_url = self.seq_indexer(seq_num, next_img_idx)
        img2 = Image.open(img2_url)

        if self.image_transform:
            img1 = self.image_transform(img1)
            img2 = self.image_transform(img2)
        
        return (img1, img2)

    def seq_numbers(self):
        return seq_numbers_from(self.seq_dirs)

    def indexer(self, idx) -> tuple:
        len_images = [len(images_in_sequence(seq)) for seq in self.seq_dirs]
        cumulative_len_images = list(itertools.accumulate(len_images))
        res = [(i, idx - l) for (i,l) in enumerate(cumulative_len_images)]
        
        for (index, diff) in res:
            if (index == 0) and is_neg(diff): #first 
                seq = self.seq_numbers()[index]
                image_num = idx
            elif is_neg(diff):
                seq = self.seq_numbers()[index]
                prev_entry = res[index-1]
                image_num = prev_entry[1]
            elif diff == 0: #last
                seq = self.seq_numbers()[index]
                prev_entry = res[index-1]
                image_num = prev_entry[1] - 1
            else:
                continue
            
            return (seq, image_num)

    def seq_indexer(self, seq_num, img_num) -> str:
        seq_path = build_seq_path(self.root_dir, seq_num)
        image_nums = images_in_sequence(seq_path)
        return os.path.join(seq_path, "img1", image_nums[img_num])

def images_in_sequence(seq_dir):
    imgs = os.listdir(os.path.join(seq_dir, "img1"))
    return sorted(imgs)

def unique_sequences_from(dir):
    numerous_seq = [f"{dir}/{x}" for x in os.listdir(dir)]
    numerous_seq = list(filter(os.path.isdir, numerous_seq))
    seq_names = seq_numbers_from(numerous_seq)
    unique_seq = list(set(seq_names))
    # use FRCNN as the default detection type
    sequences = [os.path.join(dir,f"MOT17-{s}-FRCNN") for s in unique_seq]

    return sorted(sequences)

def seq_numbers_from(dir):
    seq_folders = [os.path.split(d)[-1] for d in dir]
    seq_names = [x.split('-')[1] for x in seq_folders]
    return seq_names

def build_seq_path(dir: str, seq_num: int) -> str:
    return os.path.join(dir, f"MOT17-{seq_num}-FRCNN")

def is_neg(a) -> bool:
    return a < 0