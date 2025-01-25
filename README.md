# Zero-Shot Underwater Gesture Recognition (ICPR 2024)

## 👓 At a glance
This repository contains the official PyTorch implementation of our paper - [Zero-Shot Underwater Gesture Recognition](https://arxiv.org/pdf/2407.14103), a work done by Sandipan Sarma, Gundameedi Sai Ram Mohan, Hariansh Sehgal, and Arijit Sur at [Indian Institute of Technology Guwahati](https://www.iitg.ac.in/cse/). The work has been recently published in the proceedings of the [27th International Conference on Pattern Recognition (ICPR) 2024](https://icpr2024.org/).

![zsugr](https://github.com/user-attachments/assets/20f5ee78-6c3a-4944-a3cf-89921992b546)

## 📁 Dataset and zero-shot splits
Download the gesture recognition images for the [CADDY dataset](http://www.caddian.eu//CADDY-Underwater-Gestures-Dataset.html) inside ```raw_dataset_caddy```. The directory structure should look as follows:
```
datasets 
│
└───biograd-A
│   └───true_negatives
│   └───true_positives
│       └───raw
│           └───biograd-A_00000_left.jpg
│           └───biograd-A_00000_right.jpg
│           └───...
└───biograd-B
└───biograd-C
└───brodarski-A
└───brodarski-B
└───brodarski-C
└───brodarski-D
└───genova-A
└───splits
│   └───test_seen_random_1.csv
│   └───...

```
**Note**: Due to some file corruption, we had to rename some of the split files during experimentation. The split numbered 1, 2, and 3 in the paper correspond to the .csv files ending with 1, 4, and 5, respectively.

The dependencies can be installed by creating an Anaconda environment using zsugr.yml in the following command:

```bash
conda env create -f zsugr.yml
conda activate zsgr
```

## 🚄 Training transformer
```bash
cd scripts
sh train_random.sh
```
In ``train_random.sh``, a few important arguments that need explanation are:
- ``method``: Should always be set to "ours"
- ``our_method_type``: Should always be set to "GCAT" (corresponds to the entire proposed framework)
- ``root``: Should be set to your own root directory where you clone this repository
- ``split``: Set accordingly (can be 1, 4, or 5)
- ``split_type``: Should always be set to "random"
- ``setting``: Used for creating files with a unique name and helpful for saving logs of different versions of the framework you try. Set it as you like. We last kept it as "lr_1e-5_3dec_withLN".

## 📤 Extract gesture features 
After the transformer is trained, manually make a folder called data in your ``root`` directory, and inside it, make a folder with the dataset name (currently CADDY). Make corresponding changes in line 22 of preprocessing_gcat.py. Then run the following script:
```bash
cd scripts
sh extract_gact_features.sh
```

## 🔍 GAN training and zero-shot gesture recognition
The following command will run the train-test part of our GAN:
```bash
cd scripts
sh train_GAN.sh
```
Note: The argument ``our_method_type`` should be set to "GAN".

# :gift: Citation
If you use our work for your research, kindly star :star: our repository and consider citing our work using the following BibTex:
```
@inproceedings{sarma2025zero,
  title={Zero-Shot Underwater Gesture Recognition},
  author={Sarma, Sandipan and Sai Ram Mohan, Gundameedi and Sehgal, Hariansh and Sur, Arijit},
  booktitle={International Conference on Pattern Recognition},
  pages={346--361},
  year={2025},
  organization={Springer}
}
```
