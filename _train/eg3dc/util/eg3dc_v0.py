



import sys
# sys.path.append('./eg3dc')
sys.path = [
    p for p in sys.path
    if p not in ['./eg3dc', './_train/eg3dc/src']
] + ['./_train/eg3dc/src', ]


from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
from _util.twodee_v1 import * ; import _util.twodee_v1 as u2d
from _util.video_v1 import * ; import _util.video_v1 as uvid

import dnnlib
import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator


def load_eg3dc_model(
        inferquery,
        reload_modules=True,
        uninitialized=False,
        force_sigmoid=False,
        depth_resolution=48*2,
        depth_resolution_importance=48*2,
        ):
    if inferquery.endswith('.pkl'):
        network_pkl = inferquery
        nickname = fnstrip(network_pkl)
    else:
        name,version,epoch = inferquery.split('-')
        version,epoch = int(version),int(epoch)
        network_pkl = f'./_train/eg3dc/runs/{name}/{version:05d}/network-snapshot-{epoch:06d}.pkl'
        nickname = f'{name}-{version:05d}-{epoch:06d}'
    with dnnlib.util.open_url(network_pkl) as fp:
        network_data = legacy.load_network_pkl(fp)
    G = network_data['G_ema'].requires_grad_(False) # type: ignore
    D = network_data['D'].requires_grad_(False) # type: ignore
    if uninitialized: reload_modules = True
    if reload_modules:
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False)#
        if not uninitialized:
            misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new
    if force_sigmoid:
        G.set_force_sigmoid(True)
    G.rendering_kwargs['depth_resolution'] = depth_resolution
    G.rendering_kwargs['depth_resolution_importance'] = depth_resolution_importance
    return Dict(
        name=nickname,
        fn=network_pkl,
        G=G,
        D=D,
    )

def quickspin(G, ws, fargs=None, image_dtype='image', n=30, progress=False, **kwargs):
    # render video
    dev = G.decoder.net[0].weight.device
    with torch.no_grad():
        spin = []
        elev = torch.zeros(1).to(dev)
        rng = torch.linspace(0,360, n)[:-1,None].to(dev)
        for azim in (tqdm(rng) if progress else rng):
            torch.manual_seed(0)
            xin = uutil.Dict({
                'elevations': elev,
                'azimuths': azim,
            })
            if ws:
                xin['ws'] = ws
            if fargs!=None:
                xin.update({k:v for k,v in fargs.items() if k not in [
                    'elevations', 'azimuths', 'fovs',
                    'camera_params', 'conditioning_params',
                    'force_rays',
                ]})
            rend = G.f(xin, **kwargs)
            spin.append(I(rend[image_dtype]))
    return spin






