Attention Transfer
==============

Code for "Paying More Attention to Attention: Improving the Performance of
Convolutional Neural Networks via Attention Transfer" <https://arxiv.org/abs/1612.03928><br>
The paper is under review as a conference submission at ICLR2017: https://openreview.net/forum?id=Sks9_ajex.

<img src=https://cloud.githubusercontent.com/assets/4953728/22037632/04f54a7e-dd09-11e6-9a6b-62133fbc1c29.png width=25%><img src=https://cloud.githubusercontent.com/assets/4953728/22037801/d06c526a-dd09-11e6-8986-55c69493a075.png width=75%>


What's in this repo so far:
 * Activation-based AT code for CIFAR-10 experiments
 * Code for ImageNet experiments (ResNet-18-ResNet-34 student-teacher)
 * Pretrained with activation-based AT ResNet-18

Coming:
 * grad-based AT
 * Scenes and CUB activation-based AT code

The code uses PyTorch <https://pytorch.org>. Note that the original experiments were done
using [torch-autograd](https://github.com/twitter/torch-autograd), we have so far validated that CIFAR-10 experiments are
*exactly* reproducible in PyTorch, and are in process of doing so for ImageNet (results are
very slightly worse in PyTorch, due to hyperparameters).

Another note on the implementation, for simplicity it uses
[functional](https://github.com/pytorch/pytorch/blob/master/torch/nn/functional.py)
interface instead of [modules](https://github.com/pytorch/pytorch/tree/master/torch/nn/modules),
which is not yet documented as well.


# Requrements

First install [PyTorch](https://pytorch.org), then install [tnt](https://github.com/pytorch/tnt), and run `pip`:

```
pip install -r requirements.txt
```

You will also need OpenCV with Python bindings installed.

# Experiments

## CIFAR-10

First, train teachers:


## ImageNet

Download pretrained weights for ResNet-34:

```
wget https://s3.amazonaws.com/pytorch/h5models/resnet-34-export.hkl
```

Convergence plot:

<img width=50% src=https://cloud.githubusercontent.com/assets/4953728/22037957/5f9d493a-dd0a-11e6-9c68-8410a8c3c334.png>
