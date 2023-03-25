


try:
    import _util.util_v1 as uutil
except:
    pass


try:
    import torch
    import torch.nn as nn
except:
    pass

try:
    import torchvision as tv
    import torchvision.transforms as TT
    import torchvision.transforms.functional as TF
except:
    pass

try:
    import pytorch_lightning as pl
    import pytorch_lightning.strategies as _
    import torchmetrics
    import torchmetrics.image
except:
    pass

try:
    import wandb
except:
    pass

try:
    import kornia
except:
    pass

try:
    import einops
    from einops.layers import torch as _
except:
    pass

try:
    import cupy
except:
    pass

try:
    import lpips
except:
    pass

try:
    from addict import Dict
except:
    Dict = dict


#################### UTILITIES ####################


import contextlib, threading
@contextlib.contextmanager
def torch_seed(seed):
    _torch_seed_lock.acquire()
    state = torch.get_rng_state()
    was_det = torch.backends.cudnn.deterministic
    if seed!=None and not isinstance(seed, int):
        seed = zlib.adler32(bytes(seed, encoding='utf-8'))
    elif seed==None:
        seed = torch.seed()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    try:
        yield
    finally:
        torch.set_rng_state(state)
        torch.backends.cudnn.deterministic = was_det
        _torch_seed_lock.release()
_torch_seed_lock = threading.Lock()

def torch_seed_all(seed):
    _torch_seed_lock.acquire()
    if seed!=None and not isinstance(seed, int):
        seed = zlib.adler32(bytes(seed, encoding='utf-8'))
    elif seed==None:
        seed = torch.seed()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    _torch_seed_lock.release()
    return


# @cupy.memoize(for_each_device=True)
def cupy_launch(func, kernel):
    return cupy.cuda.compile_with_cache(kernel).get_function(func)

def reset_parameters(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    return model

def default_collate(items, device=None):
    return to(Dict(torch.utils.data.dataloader.default_collate(items)), device)

def to(x, device):
    if device is None:
        return x
    if issubclass(x.__class__, dict):
        return Dict({
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k,v in x.items()
        })
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, np.ndarray):
        return torch.tensor(x).to(device)
    assert 0, 'data not understood'


try:
    class IdentityModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            return
        def forward(self, x, *args, **kwargs):
            return x
    class Tanh10Module(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            return
        def forward(self, x, *args, **kwargs):
            return 10*torch.tanh(x/10)
except: pass


def tsuma(t, f=' 2.04f'):
    with torch.no_grad():
        s = f' shape ({",".join([str(i) for i in t.shape])})\n'
        return s+str(uutil.Table([
            # ['shape::l', ('('+','.join([str(i) for i in t.shape])+')', 'r'), ],
            ['min::l', (t.min().item(), f'r:{f}'), ' ', 'std::l', (t.std().item(), f'r:{f}'), ],
            ['mean::l', (t.mean().item(), f'r:{f}'), ' ', 'norm::l', (t.norm().item(), f'r:{f}'), ],
            ['max::l', (t.max().item(), f'r:{f}'), ],
            # ['std::l', (t.std().item(), f'r:{f}'), ],
            # ['norm::l', (t.norm().item(), f'r:{f}'), ],
        ]))+'\n'


#################### METRICS ####################

class LPIPSLoss(nn.Module):
    def __init__(self, net_type='alex', **kwargs):
        super().__init__()
        self.net_type = net_type
        assert self.net_type in ['alex', 'vgg', 'squeeze']
        self.model = lpips.LPIPS(net=self.net_type, **kwargs)
        return
    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        ans = self.model(preds, target).mean((1,2,3))
        return ans

class LaplacianPyramidLoss(nn.Module):
    def __init__(self, n_levels=3, colorspace=None, mode='l1'):
        super().__init__()
        self.n_levels = n_levels
        self.colorspace = colorspace
        self.mode = mode
        assert self.mode in ['l1', 'l2']
        return
    def forward(self, preds, target, force_levels=None, force_mode=None):
        if self.colorspace=='lab':
            preds = kornia.color.rgb_to_lab(preds.float())
            target = kornia.color.rgb_to_lab(target.float())
        lvls = self.n_levels if force_levels==None else force_levels
        preds = kornia.geometry.transform.build_pyramid(preds, lvls)
        target = kornia.geometry.transform.build_pyramid(target, lvls)
        mode = self.mode if force_mode==None else force_mode
        if mode=='l1':
            ans = torch.stack([
                (p-t).abs().mean((1,2,3))
                for p,t in zip(preds,target)
            ]).mean(0)
        elif mode=='l2':
            ans = torch.stack([
                (p-t).norm(dim=1, keepdim=True).mean((1,2,3))
                for p,t in zip(preds,target)
            ]).mean(0)
        else:
            assert 0
        return ans

def binclass_metrics(pred, gt, dims=(1,2,3)):
    assert pred.dtype==torch.bool
    assert gt.dtype==torch.bool
    
    tp = (pred & gt).sum(dims)
    acc = (pred==gt).float().mean(dims)
    pre = tp / pred.sum(dims)
    rec = tp / gt.sum(dims)
    f1 = 2*pre*rec/(pre+rec)

    pzero = pred.sum(dims)==0
    gzero = gt.sum(dims)==0
    pneqg = pzero!=gzero
    pgeqz = pzero & gzero
    f1[pneqg | ((pre==0)&(rec==0))] = 0
    pre[pneqg] = 0
    rec[pneqg] = 0
    pre[pgeqz] = 1
    rec[pgeqz] = 1
    f1[pgeqz] = 1

    return Dict({
        'f1': f1,
        'precision': pre,
        'recall': rec,
        'accuracy': acc,
    })




