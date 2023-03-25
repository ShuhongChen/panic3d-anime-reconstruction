




from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
from _util.twodee_v1 import * ; import _util.twodee_v1 as u2d
from _util.threedee_v0 import * ; import _util.threedee_v0 as u3d
from _util.video_v1 import * ; import _util.video_v1 as uvid
import _train.eg3dc.util.eg3dc_v0 as ueg3d
# import _train.eg3d.util.eg3d_v0 as ueg3d
# import _train.eg3d.util.pti_v0 as upti
from _databacks import lustrous_gltf_v0 as uvrm

import dnnlib
import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator

device = torch.device('cuda')


# exec(read('./hack/util/gltf_v0.py'))
def add_box(plot, p0, p1):
    p0x,p0y,p0z = p0
    p1x,p1y,p1z = p1
    p = plot
    a = np.asarray([p0x,p0y,p0z])
    b = np.asarray([p0x,p0y,p1z])
    c = np.asarray([p0x,p1y,p1z])
    d = np.asarray([p1x,p1y,p1z])
    e = np.asarray([p1x,p1y,p0z])
    f = np.asarray([p1x,p0y,p0z])
    g = np.asarray([p1x,p0y,p1z])
    h = np.asarray([p0x,p1y,p0z])
    p.add_lines(a,b)
    p.add_lines(b,c)
    p.add_lines(c,d)
    p.add_lines(d,e)
    p.add_lines(e,f)
    p.add_lines(f,g)
    p.add_lines(g,b)
    p.add_lines(a,h)
    p.add_lines(e,h)
    p.add_lines(a,f)
    p.add_lines(c,h)
    p.add_lines(d,g)
    return p
def add_axes(plot, scale=1, loc=[0,0,0]):
    p = plot
    loc = np.asarray(loc)
    p.add_lines(loc+scale*np.array([0,0,0]), loc+scale*np.array([1,0,0]))
    p.add_lines(loc+scale*np.array([1,0.01,-0.01]), loc+scale*np.array([1,-0.01,0.01]))
    p.add_lines(loc+scale*np.array([0,0,0]), loc+scale*np.array([0,1,0]))
    p.add_lines(loc+scale*np.array([0.01,1,-0.01]), loc+scale*np.array([-0.01,1,0.01]))
    p.add_lines(loc+scale*np.array([0.01,1.01,-0.01]), loc+scale*np.array([-0.01,1.01,0.01]))
    p.add_lines(loc+scale*np.array([0,0,0]), loc+scale*np.array([0,0,1]))
    p.add_lines(loc+scale*np.array([0.01,-0.01,1]), loc+scale*np.array([-0.01,0.01,1]))
    p.add_lines(loc+scale*np.array([0.01,-0.01,1.01]), loc+scale*np.array([-0.01,0.01,1.01]))
    p.add_lines(loc+scale*np.array([0.01,-0.01,1.02]), loc+scale*np.array([-0.01,0.01,1.02]))
    return p

def sigma2density(sigma):
    dens = sigma
    dens = nn.functional.softplus(dens - 1)
    dens = 1 - torch.exp(-dens)
    return dens
def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size
# def get_eg3d_volume(G, ws, resolution=256, max_batch=100000, ):
def get_eg3d_volume(G, xin, resolution=256, max_batch=100000, ):
    shape_res = resolution
    # max_batch = 100000
    truncation_psi = 1.0
    truncation_cutoff = None

    # run one synth due to cuda init bug
    with torch.no_grad():
        xin_ = Dict({
            # 'ws': ws,
            **xin,
            'elevations': torch.zeros(1).to(device),
            'azimuths': torch.zeros(1).to(device),
        })
        out = G.f(xin_)
        ws = xin_['ws']

    samples, voxel_origin, voxel_size = create_samples(
        N=shape_res,
        voxel_origin=[0, 0, 0],
        cube_length=G.rendering_kwargs['box_warp'] * 1,
    )#.reshape(1, -1, 3)
    # samples = samples.to(z.device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1))#, device=z.device)
    rgbs = torch.zeros((samples.shape[0], samples.shape[1], 32))#, device=z.device)
    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], max_batch, 3), device=device)
    transformed_ray_directions_expanded[..., -1] = -1

    head = 0
    # with tqdm(total = samples.shape[1]) as pbar:
    with torch.no_grad():
        while head < samples.shape[1]:
            torch.manual_seed(0)
            # out = G.sample(
            #     samples[:, head:head+max_batch].to(device),
            #     transformed_ray_directions_expanded[:, :samples.shape[1]-head],
            #     z,
            #     conditioning_params,
            #     truncation_psi=truncation_psi,
            #     truncation_cutoff=truncation_cutoff,
            #     noise_mode='const',
            # )
            smp = samples[:, head:head+max_batch].to(device)
            # trde = transformed_ray_directions_expanded[:, :samples.shape[1]-head]
            trde = -smp/smp.norm(dim=-1, keepdim=True).clip(0.01)
            # trde.zero_()
            out = G.sample_mixed(
                smp,
                trde,
                ws,
                xin['cond'],
                truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff,
                noise_mode='const',
            )
            sigmas[:, head:head+max_batch] = out['sigma'].cpu()
            rgbs[:, head:head+max_batch] = out['rgb'].cpu()
            head += max_batch
            # pbar.update(max_batch)

    # calc densities, cull/crop if needed
    densities = sigma2density(sigmas)
    triplane_crop = xin['triplane_crop'] if 'triplane_crop' in xin else None
    cull_clouds = xin['cull_clouds'] if 'cull_clouds' in xin else None
    if triplane_crop!=None:
        cropmask = triplane_crop_mask(samples, triplane_crop, G.rendering_kwargs['box_warp'])
        densities[cropmask] = -1e3
    if cull_clouds!=None:
        ccmask = cull_clouds_mask(densities, cull_clouds)
        densities[ccmask] = -1e3

    # reshape everything
    densities = densities.reshape(
        (samples.shape[0], shape_res, shape_res, shape_res, 1)
    ).flip(dims=(1,)).permute(0,-1,1,2,3)
    sigmas = sigmas.reshape(
        (samples.shape[0], shape_res, shape_res, shape_res, 1)
    ).flip(dims=(1,)).permute(0,-1,1,2,3)
    rgbs = rgbs.reshape(
        (samples.shape[0], shape_res, shape_res, shape_res, 32)
    ).flip(dims=(1,)).permute(0,-1,1,2,3)
    samples = samples.reshape(
        (samples.shape[0], shape_res, shape_res, shape_res, 3)
    ).flip(dims=(1,)).permute(0,-1,1,2,3)
    return Dict({
        'coordinates': samples,
        'sigmas': sigmas,
        'rgbs': rgbs,
        'densities': densities,
    })
# vol = get_eg3d_volume(G, xin)

def marching_cubes(vol, rgbs, boxwarp, level=0.5):
    # vol = densities.cpu().numpy()[0,0]
    # level = 0.5
    shape_res = vol.shape[-1]
    assert vol.shape[0]==vol.shape[1]==vol.shape[2]
    mc_vert, mc_face, mc_norm, mc_val = skimage.measure.marching_cubes(
        vol, level=level, spacing=(1.0, 1.0, 1.0),
        gradient_direction='descent', step_size=1,
        allow_degenerate=False, method='lewiner',
        mask=None,
    )
    # q = rgbs.cpu().numpy()
    mc_color = np.stack([
        rgbs[:3,a,b,c]
        for a,b,c in mc_vert.astype(int)
    ])
    bw = boxwarp
    mc_vert = mc_vert / shape_res * bw - 0.5*bw
    return Dict({
        'verts': mc_vert,
        'faces': mc_face,
        'normals': mc_norm,
        'values': mc_val,
        'colors': mc_color,
    })
# mc = marching_cubes(
#     vol['densities'].cpu().numpy()[0,0],
#     vol['rgbs'].cpu().numpy()[0,:3],
#     G.rendering_kwargs['box_warp'],
# )






def triplane_crop_mask(xyz_unformatted, thresh, boxwarp, allow_bottom=True):
    bw,tc = boxwarp, thresh
    device = xyz_unformatted.device
    # xyz = 0.5 * (xyz_unformatted+1) * torch.tensor([-1,1,-1]).to(device)[None,None,:]
    xyz = (xyz_unformatted) * torch.tensor([-1,1,-1]).to(device)[None,None,:]
    ans = (xyz[:,:,[0,2]].abs() <= (bw/2-tc)).all(dim=-1,keepdim=True)
    if allow_bottom:
        ans = ans | (
            (xyz[:,:,1:2] <= -(bw/2-tc)) &
            (xyz[:,:,[0,2]].abs() <= (bw/2-tc)).all(dim=-1,keepdim=True)
        )
    return ~ans
def cull_clouds_mask(denities, thresh):
    denities = torch.nn.functional.softplus(denities - 1) # activation bias of -1 makes things initialize better
    alpha = 1 - torch.exp(-denities)
    return alpha < thresh


