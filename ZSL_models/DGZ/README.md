[![arXiv](https://img.shields.io/badge/stat.ML-arXiv%3A2204.11280-B31B1B.svg)](https://arxiv.org/abs/2204.11280)
[![Framework](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/git/git-scm.com/blob/main/MIT-LICENSE.txt)

The official implementation of the **AAAI2023** paper:

<div align="center">
<h1>
<b>
Deconstructed Generation-Based Zero-Shot Model
</b>
</h1>
</div>

<div align="center">
Dubing Chen, Yuming Shen, Haofeng Zhang, Philip H.S. Torr
</div>

## Dependencies
Codes released in this work are trained and tested on:
- Ubuntu Linux
- Python 3.8.15
- Pytorch 1.13.0
- NVIDIA CUDA 11.6
- 1x NVIDIA GeForce rtx 2080 ti GPU
## Prerequisites
- Dataset: Please download the [dataset](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly), and change `--dataroot` in `config.py` to your local path. Please refer to [SDGZSL](https://github.com/uqzhichen/SDGZSL) for the finetuned features.
- Attribute: The attributes for AWA2, SUN, and APY are available in the datasets. Please download the 1024-D [CUB semantic](https://github.com/Hanzy1996/CE-GZSL) and save it to the data path.
## Train and Test
Please run the scripts in `./scripts` to reproduce the results in the paper, e.g.,
```
sh ./scripts/AWA2.sh
```
## Citation
If you recognize our work, please cite:  
```
@inproceedings{chen2023deconstructed,
            title={Deconstructed Generation-Based Zero-Shot Model},
            author={Chen, Dubing and Shen, Yuming and Zhang, Haofeng and Torr, Philip H.S.},
            booktitle={Proceedings of the 37th AAAI Conference on Artificial Intelligence (AAAI-23)},
            year={2023}
          }
```
## Acknowledgment
Our implementation is inspired by [f-CLSWGAN](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/feature-generating-networks-for-zero-shot-learning). We appreciate the authors for sharing it as an open source.

