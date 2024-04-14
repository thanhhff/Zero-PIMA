# Zero-PIMA | Zero-shot Pill-Prescription Matching with Graph Convolutional Network and Contrastive Learning

This repository is the official implementation of 'Zero-shot Pill-Prescription Matching with Graph Convolutional Network and Contrastive Learning' by Trung Thanh Nguyen, Phi Le Nguyen, Yasutomo Kawanishi, Takahiro Komamizu, and Ichiro Ide.

This work was accepted for publication in IEEE Access (IF 3.9). The open-access version will be available in a few days.

---
Environment setting using [Anaconda](https://www.anaconda.com/).

```
conda create --name pima
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install pyg -c pyg
conda install -c conda-forge transformers
conda install -c conda-forge timm
conda install -c anaconda networkx
conda install -c conda-forge wandb
```

Note:
```
You need to copy roi_heads.py and replace it with the default file of Faster RCNN in your library.
```


# Acknowledgment
The computation was carried out using the General Projects on the supercomputer "Flow" with the Information Technology Center, Nagoya University.

This work was funded by Vingroup Joint Stock Company (Vingroup JSC), Vingroup, and supported by Vingroup Innovation Foundation (VINIF) under project code VINIF.2021.DA00128. This work was partly supported by JSPS KAKENHI JP21H0355.

# Contact
Trung Thanh NGUYEN - nguyent[at]cs.is.i.nagoya-u.ac.jp or thanh.nguyen.rc[at]a.riken.jp
