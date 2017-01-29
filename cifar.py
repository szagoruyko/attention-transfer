"""
    PyTorch training code for
    "Paying More Attention to Attention: Improving the Performance of
                Convolutional Neural Networks via Attention Transfer"
    https://arxiv.org/abs/1612.03928
    
    This file includes:
     * CIFAR ResNet and Wide ResNet training code which exactly reproduces
       https://github.com/szagoruyko/wide-residual-networks
     * Activation-based attention transfer
     * Knowledge distillation implementation

    2017 Sergey Zagoruyko
"""

import argparse
import os
import json
import numpy as np
import math
import cv2
from tqdm import tqdm
import pandas as pd
import torch
import torch.optim
import torch.utils.data
from torchvision import cvtransforms
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.nn.functional as F
import torchnet as tnt
from torchnet.engine import Engine
import torch.backends.cudnn as cudnn
from nested_dict import nested_dict
from collections import OrderedDict
from utils import conv_params, linear_params, bnparams, bnstats, \
        distillation, l2_normalize, cast, data_parallel

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Wide Residual Networks')
# Model options
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--width', default=1, type=float)
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--data_root', default='.', type=str)
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--nthread', default=4, type=int)
parser.add_argument('--teacher_id', default='', type=str)

# Training options
parser.add_argument('--batchSize', default=128, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--epochs', default=200, type=int, metavar='N',
        help='number of total epochs to run')
parser.add_argument('--weightDecay', default=0.0005, type=float)
parser.add_argument('--epoch_step', default='[60,120,160]', type=str,
        help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--optim_method', default='SGD', type=str)
parser.add_argument('--randomcrop_pad', default=4, type=float)
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


def create_dataset(opt, mode):
    convert = tnt.transform.compose([
        lambda x: x.astype(np.float32),
        cvtransforms.Normalize([125.3, 123.0, 113.9], [63.0,  62.1,  66.7]),
        lambda x: x.transpose(2,0,1).astype(np.float32),
        torch.from_numpy,
        ])

    train_transform = tnt.transform.compose([
        cvtransforms.RandomHorizontalFlip(),
        cvtransforms.Pad(opt.randomcrop_pad, cv2.BORDER_REFLECT),
        cvtransforms.RandomCrop(32),
        convert,
        ])

    ds = getattr(datasets, opt.dataset)(opt.data_root, train=mode, download=True)
    smode = 'train' if mode else 'test'
    ds = tnt.dataset.TensorDataset([
        getattr(ds, smode+'_data').transpose(0,2,3,1),
        getattr(ds, smode+'_labels')])
    return ds.transform({0: train_transform if mode else convert})


def resnet(depth, width, num_classes):
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = int((depth - 4) / 6)
    widths = np.floor(np.asarray([16.,32.,64.]) * width).astype(np.int)

    def gen_block_params(ni, no):
        return {
                'conv0': conv_params(ni, no, 3),
                'conv1': conv_params(no, no, 3),
                'bn0': bnparams(ni),
                'bn1': bnparams(no),
                'convdim': conv_params(ni, no, 1) if ni != no else None,
                }

    def gen_group_params(ni, no, count):
        return {'block%d'%i: gen_block_params(ni if i==0 else no, no)
                for i in range(count)}

    def gen_group_stats(ni, no, count):
        return {'block%d'%i: {'bn0': bnstats(ni if i==0 else no), 'bn1': bnstats(no)}
                for i in range(count)}

    params = nested_dict({
            'conv0': conv_params(3,16,3),
            'group0': gen_group_params(16, widths[0], n),
            'group1': gen_group_params(widths[0], widths[1], n),
            'group2': gen_group_params(widths[1], widths[2], n),
            'bn': bnparams(widths[2]),
            'fc': linear_params(widths[2], num_classes),
            })

    stats = nested_dict({
            'group0': gen_group_stats(16, widths[0], n),
            'group1': gen_group_stats(widths[0], widths[1], n),
            'group2': gen_group_stats(widths[1], widths[2], n),
            'bn': bnstats(widths[2]),
            })

    flat_params = OrderedDict()
    flat_stats = OrderedDict()
    for keys,v in params.iteritems_flat():
        if v is not None:
            flat_params['.'.join(keys)] = Variable(v, requires_grad=True)
    for keys,v in stats.iteritems_flat():
        flat_stats['.'.join(keys)] = v

    def activation(x, params, stats, base, mode):
        return F.relu(F.batch_norm(x, weight = params[base+'.weight'],
                                   bias = params[base+'.bias'],
                                   running_mean = stats[base+'.running_mean'],
                                   running_var = stats[base+'.running_var'],
                                   training = mode, momentum = 0.1, eps = 1e-5))

    def block(x, params, stats, base, mode, stride):
        o1 = activation(x, params, stats, base+'.bn0', mode)
        y = F.conv2d(o1, params[base+'.conv0'], stride=stride, padding=1)
        o2 = activation(y, params, stats, base+'.bn1', mode)
        z = F.conv2d(o2, params[base+'.conv1'], stride=1, padding=1)
        if base + '.convdim' in params:
            return z + F.conv2d(o1, params[base+'.convdim'], stride=stride)
        else:
            return z + x

    def group(o, params, stats, base, mode, stride):
        for i in range(n):
            o = block(o, params, stats, '%s.block%d'%(base,i), mode, stride if i==0 else 1)
        return o

    def f(input, params, stats, mode, prefix=''):
        x = F.conv2d(input, params[prefix+'conv0'], padding=1)
        g0 = group(x, params, stats, prefix+'group0', mode, 1)
        g1 = group(g0, params, stats, prefix+'group1', mode, 2)
        g2 = group(g1, params, stats, prefix+'group2', mode, 2)
        o = activation(g2, params, stats, prefix+'bn', mode)
        o = F.avg_pool2d(o, 8, 1, 0)
        o = o.view(o.size(0), -1)
        o = F.linear(o, params[prefix+'fc.weight'], params[prefix+'fc.bias'])
        return o, [g0, g1, g2]

    return f, flat_params, flat_stats


def at(x):
    q = x.pow(2).mean(1).view(x.size(0), -1)
    return l2_normalize(q)

def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()


def main():
    opt = parser.parse_args()
    print 'parsed options:', vars(opt)
    epoch_step = json.loads(opt.epoch_step)
    num_classes = 10 if opt.dataset == 'CIFAR10' else 100

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    # to prevent opencv from initializing CUDA in workers
    torch.randn(8).cuda()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    def create_iterator(mode):
        ds = create_dataset(opt, mode)
        return ds.parallel(batch_size=opt.batchSize, shuffle=mode,
                           num_workers=opt.nthread, pin_memory=True)

    train_loader = create_iterator(True)
    test_loader = create_iterator(False)

    # deal with student first
    f_s, params_s, stats_s = resnet(opt.depth, opt.width, num_classes)

    # deal with teacher
    if opt.teacher_id != '':
        with open(os.path.join('logs', opt.teacher_id, 'log.txt'), 'r') as ff:
            line = ff.readline()
            r = line.find('json_stats')
            info = json.loads(line[r+12:])
        f_t = resnet(info['depth'], info['width'], num_classes)[0]
        model_data = torch.load(os.path.join('logs', opt.teacher_id, 'model.pt7'))
        params_t = model_data['params']
        stats_t = model_data['stats']


        # merge teacher and student params and stats
        params = {'student.'+k: v for k, v in params_s.iteritems()}
        for k, v in params_t.iteritems():
            v.requires_grad = False
            params['teacher.'+k] = v
        stats = {'student.'+k: v for k, v in stats_s.iteritems()}
        stats.update({'teacher.'+k: v for k, v in stats_t.iteritems()})

        def f(inputs, params, stats, mode):
            y_s, g_s = f_s(inputs, params, stats, mode, 'student.')
            y_t, g_t = f_t(inputs, params, stats, False, 'teacher.')
            return y_s, y_t, [at_loss(x, y) for x,y in zip(g_s, g_t)]
    else:
        f, params, stats = f_s, params_s, stats_s

    optimizable = [v for v in params.itervalues() if v.requires_grad]

    def create_optimizer(opt, lr):
        print 'creating optimizer with lr = ', lr
        if opt.optim_method == 'SGD':
            return torch.optim.SGD(optimizable, lr, 0.9, weight_decay=opt.weightDecay)
        elif opt.optim_method == 'Adam':
            return torch.optim.Adam(optimizable, lr)

    optimizer = create_optimizer(opt, opt.lr)

    epoch = 0
    if opt.resume != '':
        state_dict = torch.load(opt.resume)
        epoch = state_dict['epoch']
        params, stats = state_dict['params'], state_dict['stats']
        optimizer.load_state_dict(state_dict['optimizer'])

    print '\nParameters:'
    print pd.DataFrame([(key, v.size(), torch.typename(v.data)) for key,v in params.items()])
    print '\nAdditional buffers:'
    print pd.DataFrame([(key, v.size(), torch.typename(v)) for key,v in stats.items()])

    n_parameters = sum([p.numel() for p in params_s.values() + stats_s.values()])
    print '\nTotal number of parameters:', n_parameters

    meter_loss = tnt.meter.AverageValueMeter()
    classacc = tnt.meter.ClassErrorMeter(accuracy=True)
    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')
    meters_at = [tnt.meter.AverageValueMeter() for i in range(3)]

    if not os.path.exists(opt.save):
        os.mkdir(opt.save)

    def h(sample):
        inputs = Variable(cast(sample[0], opt.dtype))
        targets = Variable(cast(sample[1], 'long'))
        if opt.teacher_id != '':
            y_s, y_t, loss_groups = data_parallel(f, inputs, params, stats, sample[2], np.arange(opt.ngpu))
            loss_groups = [v.sum() for v in loss_groups]
            [m.add(v.data[0]) for m,v in zip(meters_at, loss_groups)]
            return distillation(y_s, y_t, targets, opt.temperature, opt.alpha) \
                    + opt.beta * sum(loss_groups), y_s
        else:
            y = data_parallel(f, inputs, params, stats, sample[2], np.arange(opt.ngpu))[0]
            return F.cross_entropy(y, targets), y

    def log(t):
        torch.save(dict(params=params, stats=stats,
                        optimizer=optimizer.state_dict(), epoch=t['epoch']),
                open(os.path.join(opt.save, 'model.pt7'), 'w'))
        z = vars(opt).copy(); z.update(t)
        logname = os.path.join(opt.save, 'log.txt')
        with open(logname, 'a') as f:
            f.write('json_stats: ' + json.dumps(z) + '\n')
        print z

    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        classacc.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].data[0])

    def on_start(state):
        state['epoch'] = epoch

    def on_start_epoch(state):
        classacc.reset()
        meter_loss.reset()
        timer_train.reset()
        [meter.reset() for meter in meters_at]
        state['iterator'] = tqdm(train_loader)

        epoch = state['epoch'] + 1
        if epoch in epoch_step:
            lr = state['optimizer'].param_groups[0]['lr']
            optimizer = create_optimizer(opt, lr * opt.lr_decay_ratio)

    def on_end_epoch(state):
        train_loss = meter_loss.value()
        train_acc = classacc.value()
        train_time = timer_train.value()
        meter_loss.reset()
        classacc.reset()
        timer_test.reset()

        engine.test(h, test_loader)
        
        test_acc = classacc.value()[0]
        print log({
            "train_loss": train_loss[0],
            "train_acc": train_acc[0],
            "test_loss": meter_loss.value()[0],
            "test_acc": test_acc,
            "epoch": state['epoch'],
            "num_classes": num_classes,
            "n_parameters": n_parameters,
            "train_time": train_time,
            "test_time": timer_test.value(),
            "at_losses": [m.value() for m in meters_at],
           })
        print '==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m' % \
                       (opt.save, state['epoch'], opt.epochs, test_acc)

    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_start'] = on_start
    engine.train(h, train_loader, opt.epochs, optimizer) 

if __name__ == '__main__':
    main()
