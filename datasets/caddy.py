import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.io import read_image
from PIL import Image
from torch.utils.data import DataLoader
import numpy as np
# import datasets.transforms as T
from pathlib import Path
import clip

# label id to label name mapping 
label_dict =    {
    0:  "start_comm",
    1:  "end_comm",
    2:  "up",
    3:  "down",
    4:  "photo",
    5:  "backwards",
    6:  "carry",
    7:  "boat",
    8:  "here",
    9:  "mosaic",
    10: "num_delimiter",
    11: "one",
    12: "two",
    13: "three",
    14: "four",
    15: "five",
    # 16: "nothing"
}


# seen_csv_file_path='seen_img_label.csv'
# unseen_csv_file_path='unseen_img_label.csv'
# img_dir='/raw_dataset_caddy'

class CaddyDataset(Dataset):
    def __init__(self, img_set, img_folder, anno_file, args, transforms=None):
        self.img_set = img_set
        self.img_folder = img_folder
        self.anno_file = anno_file
   
        self.img_labels = pd.read_csv(self.anno_file)

        # self.clip_feats_folder = clip_feats_folder

        self._transforms = transforms
        if self.img_set == 'train':
            print("train dataset object created")

        else:
            print(" test dataset object created")

        # Add a new attribute to store image paths
        self.image_paths = [os.path.join(self.img_folder, img) for img in self.img_labels.iloc[:, 0]]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _, self.clip_preprocess = clip.load(args.clip_version, device)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # print(self.img_labels.iloc[idx, 0][1:])
        img_path = os.path.join(self.img_folder, self.img_labels.iloc[idx, 0][1:])
        #absolute path is required
        # print("img_path-----",img_path)
        
        # img_path = os.getcwd()+img_path
        img = Image.open(img_path).convert('RGB')
        # print(" = = image === ", img)
        w, h = img.size
        label = self.img_labels.iloc[idx, 1]
        # print("Label------:", label)
        
        target = {}
        target['size'] = torch.as_tensor([int(h), int(w)])
        target['label'] = label
        # print("\n = = = checkpoint 1, target from __getitem__: = = =  \n",target)
        if self._transforms is not None:
            img_0 = self._transforms[0](img)
            img_final = self._transforms[1](img_0)
            # img_0, target_0 = self._transforms[0](img, target)
            # img, target = self._transforms[1](img_0, target_0)
        # img_0 is unnormalized transform output
        
        # print("\n = = =  checkpoint 2  = = = \n")
        clip_inputs = self.clip_preprocess(img)
        # clip_inputs = self.clip_preprocess(img)
        target['clip_inputs'] = clip_inputs      
        # target['filename'] = img_anno['file_name']
        # print("\n += = =  checkpoint 3  = = = target: \n",target)
        return img_final, target

        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        # return image, label
    

def make_caddy_transforms(image_set):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    # if image_set == 'train':
    #     return [T.Compose([
    #         T.RandomHorizontalFlip(),
    #         T.ColorJitter(.4, .4, .4),
    #         T.RandomSelect(
    #             T.RandomResize(scales, max_size=1333),
    #             T.Compose([
    #                 T.RandomResize([400, 500, 600]),
    #                 T.RandomSizeCrop(384, 600),
    #                 T.RandomResize(scales, max_size=1333),
    #             ]))]
    #         ),
    #         normalize
    #         ]

    if image_set == 'train':
        return [T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(45)
            ]
            ),
            normalize
            ]
    if image_set == 'test_seen' or image_set == 'test_unseen':
        return [T.Compose([
            T.Resize(256),
            T.CenterCrop(224)
        ]),
            normalize
        ]

    # if image_set == 'train':
    #     return [normalize]
    # if image_set == 'val':
    #     return [normalize]

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.root)
    assert root.exists(), f'provided data path {root} does not exist'
    PATHS = {
        'train': (root / args.image_folder, args.train_path),
        'test_seen': (root / args.image_folder, args.test_seen_path),
        'test_unseen': (root / args.image_folder, args.test_unseen_path)
    }

    # img_folder, anno_file, clip_feats_folder = PATHS[image_set]
    img_folder, anno_file = PATHS[image_set]
    # print(img_folder)


    dataset = CaddyDataset(image_set, img_folder, anno_file, args=args, transforms=make_caddy_transforms(image_set))

    return dataset    




