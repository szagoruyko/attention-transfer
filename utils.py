import math
import torch
import torch.cuda.comm as comm
from torch.nn.parallel._functions import Broadcast
from torch.nn.parallel import scatter, parallel_apply, gather
import torch.nn.functional as F

def cast(params, dtype='float'):
    if isinstance(params, dict):
        return {k: cast(v, dtype) for k,v in params.items()}
    else:
        return getattr(params.cuda(), dtype)()
        

def conv_params(ni,no,k=1,g=1):
    assert ni % g == 0
    return cast(torch.Tensor(no,ni/g,k,k).normal_(0,2/math.sqrt(ni*k*k)))

def linear_params(ni,no):
    return cast(dict(
        weight= torch.Tensor(no,ni).normal_(0,2/math.sqrt(ni)),
        bias= torch.zeros(no)))

def bnparams(n):
    return cast(dict(
        weight= torch.Tensor(n).uniform_(),
        bias=   torch.zeros(n)))

def bnstats(n):
    return cast(dict(
        running_mean= torch.zeros(n),
        running_var=  torch.ones(n)))

def data_parallel(f, input, params, stats, mode, device_ids, output_device=None):
    if output_device is None:
        output_device = device_ids[0]

    if len(device_ids) == 1:
        return f(input, params, stats, mode)

    def replicate(param_dict, g):
        replicas = [{} for d in device_ids]
        for k,v in param_dict.iteritems():
            for i,u in enumerate(g(v)):
                replicas[i][k] = u
        return replicas

    params_replicas = replicate(params, lambda x: Broadcast(device_ids)(x))
    stats_replicas = replicate(stats, lambda x: comm.broadcast(x, device_ids))

    replicas = [lambda x,p=p,s=s,mode=mode: f(x,p,s,mode)
            for i,(p,s) in enumerate(zip(params_replicas, stats_replicas))]
    inputs = scatter(input, device_ids)
    outputs = parallel_apply(replicas, inputs)
    return gather(outputs, output_device)


def distillation(y, teacher_scores, labels, T, alpha):
    return F.kl_div(F.log_softmax(y/T), F.softmax(teacher_scores/T)) * (T*T * 2. * alpha) \
            + F.cross_entropy(y, labels) * (1. - alpha)

def l2_normalize(x, dim=1, epsilon=1e-12):
    return x * x.pow(2).sum(dim).clamp(min=epsilon).rsqrt().expand_as(x)

