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


## More updates: Coming soon!
