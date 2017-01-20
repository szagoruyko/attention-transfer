Attention Transfer
==============

PyTorch code for "Paying More Attention to Attention: Improving the Performance of
Convolutional Neural Networks via Attention Transfer" <https://arxiv.org/abs/1612.03928><br>
The paper is under review as a conference submission at ICLR2017: https://openreview.net/forum?id=Sks9_ajex

<img src=https://cloud.githubusercontent.com/assets/4953728/22037632/04f54a7e-dd09-11e6-9a6b-62133fbc1c29.png width=25%><img src=https://cloud.githubusercontent.com/assets/4953728/22037801/d06c526a-dd09-11e6-8986-55c69493a075.png width=75%>


What's in this repo so far:
 * Activation-based AT code for CIFAR-10 experiments
 * Code for ImageNet experiments (ResNet-18-ResNet-34 student-teacher)

Coming:
 * grad-based AT
 * Scenes and CUB activation-based AT code
 * Pretrained with activation-based AT ResNet-18

The code uses PyTorch <https://pytorch.org>. Note that the original experiments were done
using [torch-autograd](https://github.com/twitter/torch-autograd), we have so far validated that CIFAR-10 experiments are
*exactly* reproducible in PyTorch, and are in process of doing so for ImageNet (results are
very slightly worse in PyTorch, due to hyperparameters).

bibtex:

```
@article{Zagoruyko2016AT,
    author = {Sergey Zagoruyko and Nikos Komodakis},
    title = {Paying More Attention to Attention: Improving the Performance of
             Convolutional Neural Networks via Attention Transfer},
    url = {https://arxiv.org/abs/1612.03928},
    year = {2016}}
```

## Requrements

First install [PyTorch](https://pytorch.org), then install [torchnet](https://github.com/pytorch/tnt):

```
git clone https://github.com/pytorch/tnt
cd tnt
python setup.py install
```

Install [OpenCV](https://opencv.org) with Python bindings, and `torchvision`
with OpenCV transforms:

```
git clone https://github.com/szagoruyko/vision
cd vision; git checkout opencv
python setup.py install
```

Finally, install other Python packages:

```
pip install -r requirements.txt
```

## Experiments

### CIFAR-10

This section describes how to get the results in the table 1 of the paper.

First, train teachers:

```
python cifar.py --save logs/resnet_40_1_teacher --depth 40 --width 1
python cifar.py --save logs/resnet_16_2_teacher --depth 16 --width 2
python cifar.py --save logs/resnet_40_2_teacher --depth 40 --width 2
```

To train with activation-based AT do:

```
python cifar.py --save logs/at_16_1_16_2 --teacher_id resnet_16_2_teacher --beta 1e+3
```

To train with KD:

```
python cifar.py --save logs/kd_16_1_16_2 --teacher_id resnet_16_2_teacher --alpha 0.9
```

We plan to add AT+KD with decaying `beta` to get the best knowledge transfer results soon.

### ImageNet

#### Pretrained model

We provide ResNet-18 pretrained model with activation based AT:

| Model | val error |
|:------|:---------:|
|ResNet-18 | 30.4, 10.8 |
|ResNet-18-ResNet-34-AT | 29.3, 10.0 |

Download link: [coming]

Model definition: [coming]

Convergence plot:

<img width=50% src=https://cloud.githubusercontent.com/assets/4953728/22037957/5f9d493a-dd0a-11e6-9c68-8410a8c3c334.png>

#### Train from scratch

Download pretrained weights for ResNet-34
(see also [functional-zoo](https://github.com/szagoruyko/functional-zoo) for more
information):

```
wget https://s3.amazonaws.com/pytorch/h5models/resnet-34-export.hkl
```

Prepare the data following [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch)
and run training (e.g. using 2 GPUs):

```
python imagenet.py --imagenetpath ~/ILSVRC2012 --depth 18 --width 1 \
                   --teacher_params resnet-34-export.hkl --gpu_id 0,1 --ngpu 2 \
                   --beta 1e+3
```
