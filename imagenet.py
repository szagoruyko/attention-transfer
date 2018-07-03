"""
    PyTorch training code for
    "Paying More Attention to Attention: Improving the Performance of
                Convolutional Neural Networks via Attention Transfer"
    https://arxiv.org/abs/1612.03928
    
    This file includes:
     * ImageNet ResNet training code that follows
       https://github.com/facebook/fb.resnet.torch
     * Activation-based attention transfer on ImageNet

    2017 Sergey Zagoruyko
"""

import argparse
import os
import re
import json
import numpy as np
from torch.optim import SGD
from tqdm import tqdm
import torch
import torchnet as tnt
from torchnet.engine import Engine
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torch.backends import cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import utils
from collections import OrderedDict

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Wide Residual Networks')
# Model options
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--depth', default=18, type=int)
parser.add_argument('--width', default=1, type=float)
parser.add_argument('--imagenetpath', default='/home/zagoruys/ILSVRC2012', type=str)
parser.add_argument('--nthread', default=4, type=int)
parser.add_argument('--teacher_params', default='', type=str)

# Training options
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--epoch_step', default='[30,60,90]', type=str,
                    help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.1, type=float)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--temperature', default=4, type=float)
parser.add_argument('--alpha', default=0, type=float)
parser.add_argument('--beta', default=0, type=float)


# Device options
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--save', default='', type=str,
                    help='save parameters and logs in this folder')
parser.add_argument('--ngpu', default=1, type=int,
                    help='number of GPUs to use for training')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')


def get_iterator(imagenetpath, batch_size, nthread, mode):
    imagenetpath = os.path.expanduser(imagenetpath)

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    print("| setting up data loader...")
    if mode:
        traindir = os.path.join(imagenetpath, 'train')
        ds = ImageFolder(traindir, T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ]))
    else:
        valdir = os.path.join(imagenetpath, 'val')
        ds = ImageFolder(valdir, T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize,
        ]))

    return DataLoader(ds, batch_size=batch_size, shuffle=mode,
                      num_workers=nthread, pin_memory=True)


def define_teacher(params_file):
    """ Defines student resnet
        
        Network size is determined from parameters, assuming
        pre-activation basic-block resnet (ResNet-18 or ResNet-34)
    """
    params = torch.load(params_file)

    params = {k: p.cuda() for k, p in params.items()}

    blocks = [sum([re.match('group%d.block\d+.conv0.weight'%j, k) is not None
                   for k in list(params.keys())]) for j in range(4)]

    def conv2d(input, params, base, stride=1, pad=0):
        return F.conv2d(input, params[base + '.weight'], params[base + '.bias'], stride, pad)

    def group(input, params, base, stride, n):
        o = input
        for i in range(0,n):
            b_base = '%s.block%d.conv' % (base, i)
            x = o
            o = conv2d(x, params, b_base + '0', pad=1, stride=stride if i == 0 else 1)
            o = F.relu(o, inplace=True)
            o = conv2d(o, params, b_base + '1', pad=1)
            if i == 0 and stride != 1:
                o += F.conv2d(x, params[b_base + '_dim.weight'], stride=stride)
            else:
                o += x
            o = F.relu(o, inplace=True)
        return o

    def f(inputs, params, pr=''):
        o = conv2d(inputs, params, pr+'conv0', 2, 3)
        o = F.relu(o, inplace=True)
        o = F.max_pool2d(o, 3, 2, 1)
        o_g0 = group(o, params, pr+'group0', 1, blocks[0])
        o_g1 = group(o_g0, params, pr+'group1', 2, blocks[1])
        o_g2 = group(o_g1, params, pr+'group2', 2, blocks[2])
        o_g3 = group(o_g2, params, pr+'group3', 2, blocks[3])
        o = F.avg_pool2d(o_g3, 7, 1, 0)
        o = o.view(o.size(0), -1)
        o = F.linear(o, params[pr+'fc.weight'], params[pr+'fc.bias'])
        return o, (o_g0, o_g1, o_g2, o_g3)

    return f, params


def define_student(depth, width):
    definitions = {18: [2,2,2,2],
                   34: [3,4,6,5]}
    assert depth in list(definitions.keys())
    widths = [int(w * width) for w in (64, 128, 256, 512)]
    blocks = definitions[depth]

    def gen_block_params(ni, no):
        return {'conv0': utils.conv_params(ni, no, 3),
                'conv1': utils.conv_params(no, no, 3),
                'bn0': utils.bnparams(no),
                'bn1': utils.bnparams(no),
                'convdim': utils.conv_params(ni, no, 1) if ni != no else None,
                }

    def gen_group_params(ni, no, count):
        return {'block%d'%i: gen_block_params(ni if i==0 else no, no)
                for i in range(count)}

    flat_params = OrderedDict(utils.flatten({
        'conv0': utils.conv_params(3, 64, 7),
        'bn0': utils.bnparams(64),
        'group0': gen_group_params(64, widths[0], blocks[0]),
        'group1': gen_group_params(widths[0], widths[1], blocks[1]),
        'group2': gen_group_params(widths[1], widths[2], blocks[2]),
        'group3': gen_group_params(widths[2], widths[3], blocks[3]),
        'fc': utils.linear_params(widths[3], 1000),
    }))

    utils.set_requires_grad_except_bn_(flat_params)

    def block(x, params, base, mode, stride):
        y = F.conv2d(x, params[base+'.conv0'], stride=stride, padding=1)
        o1 = F.relu(utils.batch_norm(y, params, base+'.bn0', mode), inplace=True)
        z = F.conv2d(o1, params[base+'.conv1'], stride=1, padding=1)
        o2 = utils.batch_norm(z, params, base+'.bn1', mode)
        if base + '.convdim' in params:
            return F.relu(o2 + F.conv2d(x, params[base+'.convdim'], stride=stride), inplace=True)
        else:
            return F.relu(o2 + x, inplace=True)

    def group(o, params, base, mode, stride, n):
        for i in range(n):
            o = block(o, params, '%s.block%d' % (base, i), mode, stride if i == 0 else 1)
        return o

    def f(input, params, mode, pr=''):
        o = F.conv2d(input, params[pr+'conv0'], stride=2, padding=3)
        o = F.relu(utils.batch_norm(o, params, pr+'bn0', mode), inplace=True)
        o = F.max_pool2d(o, 3, 2, 1)
        g0 = group(o, params, pr+'group0', mode, 1, blocks[0])
        g1 = group(g0, params, pr+'group1', mode, 2, blocks[1])
        g2 = group(g1, params, pr+'group2', mode, 2, blocks[2])
        g3 = group(g2, params, pr+'group3', mode, 2, blocks[3])
        o = F.avg_pool2d(g3, 7)
        o = o.view(o.size(0), -1)
        o = F.linear(o, params[pr+'fc.weight'], params[pr+'fc.bias'])
        return o, [g0, g1, g2, g3]

    return f, flat_params


def main():
    opt = parser.parse_args()
    epoch_step = json.loads(opt.epoch_step)
    print('parsed options:', vars(opt))

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    epoch_step = json.loads(opt.epoch_step)

    if not os.path.exists(opt.save):
        os.mkdir(opt.save)

    f_s, params_s = define_student(opt.depth, opt.width)
    f_t, params_t = define_teacher(opt.teacher_params)
    params = {'student.'+k: v for k, v in params_s.items()}
    params.update({'teacher.'+k: v for k, v in params_t.items()})

    params = OrderedDict((k, p.cuda().detach().requires_grad_(p.requires_grad)) for k, p in params.items())

    optimizable = [v for v in params.values() if v.requires_grad]
    def create_optimizer(opt, lr):
        print('creating optimizer with lr = ', lr)
        return SGD(optimizable, lr, momentum=0.9, weight_decay=opt.weight_decay)

    optimizer = create_optimizer(opt, opt.lr)

    iter_train = get_iterator(opt.imagenetpath, opt.batch_size, opt.nthread, True)
    iter_test = get_iterator(opt.imagenetpath, opt.batch_size, opt.nthread, False)

    epoch = 0
    if opt.resume != '':
        state_dict = torch.load(opt.resume)
        epoch = state_dict['epoch']
        params_tensors = state_dict['params']
        for k, v in params.items():
            v.data.copy_(params_tensors[k])
        optimizer.load_state_dict(state_dict['optimizer'])

    print('\nParameters:')
    utils.print_tensor_dict(params)


    n_parameters = sum(p.numel() for p in optimizable)
    print('\nTotal number of parameters:', n_parameters)

    meter_loss = tnt.meter.AverageValueMeter()
    classacc = tnt.meter.ClassErrorMeter(topk=[1, 5], accuracy=True)
    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')
    meters_at = [tnt.meter.AverageValueMeter() for i in range(4)]

    def f(inputs, params, mode):
        y_s, g_s = f_s(inputs, params, mode, 'student.')
        with torch.no_grad():
            y_t, g_t = f_t(inputs, params, 'teacher.')
        return y_s, y_t, [utils.at_loss(x, y) for x, y in zip(g_s, g_t)]

    def h(sample):
        inputs, targets, mode = sample
        inputs = inputs.cuda().detach()
        targets = targets.cuda().long().detach()
        y_s, y_t, loss_groups = utils.data_parallel(f, inputs, params, mode, range(opt.ngpu))
        loss_groups = [v.sum() for v in loss_groups]
        [m.add(v.item()) for m,v in zip(meters_at, loss_groups)]
        return utils.distillation(y_s, y_t, targets, opt.temperature, opt.alpha) \
                + opt.beta * sum(loss_groups), y_s

    def log(t, state):
        torch.save(dict(params={k: v.data for k, v in params.items()},
                        optimizer=state['optimizer'].state_dict(),
                        epoch=t['epoch']),
                   os.path.join(opt.save, 'model.pt7'))
        z = vars(opt).copy(); z.update(t)
        logname = os.path.join(opt.save, 'log.txt')
        with open(logname, 'a') as f:
            f.write('json_stats: ' + json.dumps(z) + '\n')
        print(z)

    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        classacc.add(state['output'].data, state['sample'][1])
        loss = state['loss'].item()
        meter_loss.add(loss)
        if state['train']:
            state['iterator'].set_postfix(loss=loss)

    def on_start(state):
        state['epoch'] = epoch

    def on_start_epoch(state):
        classacc.reset()
        meter_loss.reset()
        timer_train.reset()
        [meter.reset() for meter in meters_at]
        state['iterator'] = tqdm(iter_train, dynamic_ncols=True)

        epoch = state['epoch'] + 1
        if epoch in epoch_step:
            lr = state['optimizer'].param_groups[0]['lr']
            state['optimizer'] = create_optimizer(opt, lr * opt.lr_decay_ratio)

    def on_end_epoch(state):
        train_loss = meter_loss.value()
        train_acc = classacc.value()
        train_time = timer_train.value()
        meter_loss.reset()
        classacc.reset()
        timer_test.reset()

        engine.test(h, iter_test)

        print(log({
            "train_loss": train_loss[0],
            "train_acc": train_acc,
            "test_loss": meter_loss.value()[0],
            "test_acc": classacc.value(),
            "epoch": state['epoch'],
            "n_parameters": n_parameters,
            "train_time": train_time,
            "test_time": timer_test.value(),
            "at_losses": [m.value() for m in meters_at],
           }, state))

    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_start'] = on_start
    engine.train(h, iter_train, opt.epochs, optimizer)


if __name__ == '__main__':
    main()
