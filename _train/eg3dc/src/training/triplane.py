# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
from torch_utils import persistence
from training.networks_stylegan2 import Generator as StyleGAN2Backbone
from training.volumetric_rendering.renderer import ImportanceRenderer
from training.volumetric_rendering.ray_sampler import RaySampler
import dnnlib
from camera_utils import LookAtPoseSampler
from torch import nn
import kornia

import numpy as np
import _util.util_v1 as uutil
import _util.pytorch_v1 as utorch
import _util.twodee_v1 as u2d

import _databacks.lustrous_renders_v1 as dklustr


@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs = {},
        cond_mode = None,
        triplane_width=32,
        sr_channels_hidden=256,
        backbone_resolution=256,
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        self.renderer = ImportanceRenderer(use_triplane=rendering_kwargs.get('use_triplane', False))
        self.ray_sampler = RaySampler()
        self.triplane_width = triplane_width
        self.backbone_resolution = backbone_resolution
        self.backbone = StyleGAN2Backbone(
            z_dim, c_dim, w_dim, img_resolution=backbone_resolution,
            img_channels=self.triplane_width*3*(
                1 if 'triplane_depth' not in rendering_kwargs else rendering_kwargs['triplane_depth']
            ),            cond_mode=cond_mode,
            mapping_kwargs=mapping_kwargs, **synthesis_kwargs,
        )
        self.superresolution = dnnlib.util.construct_class_by_name(
            class_name=rendering_kwargs['superresolution_module'],
            channels=32,
            channels_hidden=sr_channels_hidden,
            img_resolution=img_resolution,
            sr_num_fp16_res=sr_num_fp16_res,
            sr_antialias=rendering_kwargs['sr_antialias'],
            **sr_kwargs,
        )
        self.decoder = OSGDecoder(
            self.triplane_width,
            {
                'decoder_lr_mul':
                    rendering_kwargs.get('decoder_lr_mul', 1),
                'decoder_output_dim': 32,
            },
        )
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
        self.cond_mode = cond_mode
    
        self._last_planes = None
        return
    
    def mapping(
            self, z, c, cond,
            truncation_psi=1,
            truncation_cutoff=None,
            update_emas=False,
        ):
        # print(c.dtype, c.device)
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        # shu's fine-tuning hacks
        if 'c_gen_conditioning_force_ffhq' in self.rendering_kwargs and \
                self.rendering_kwargs['c_gen_conditioning_force_ffhq']:
            intr = torch.tensor([
                [4.2647, 0.    , 0.5   ],
                [0.    , 4.2647, 0.5   ],
                [0.    , 0.    , 1.    ],
            ], dtype=c.dtype, device=c.device).flatten()[None,].repeat(c.shape[0],1)
            if self.rendering_kwargs['c_gen_conditioning_zero']:
                extr = torch.tensor([
                    [1.000,  0.000,  0.000,  0.000, ],
                    [0.000, -1.000,  0.000,  0.000, ],
                    [0.000,  0.000, -1.000,  2.700, ],
                    [0.000,  0.000,  0.000,  1.000, ],
                ], dtype=c.dtype, device=c.device).flatten()[None,].repeat(c.shape[0],1)
            else:
                extr = c.detach().clone()[:,:16].view(-1,4,4)
                t = extr[:,:3,3]
                t = t * 2.7 / t.norm(dim=1, keepdim=True)
                extr[:,:3,3] = t
                extr = extr.flatten(1)
            c = torch.cat([extr, intr], dim=1)
        # print(z.dtype, z.device)
        # print(c.dtype, c.device)
        # print(self.backbone.mapping.embed.weight.dtype, self.backbone.mapping.embed.weight.device)
        return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), cond, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
    def mapping_zplus(
            self, zs, c, cond,
            truncation_psi=1,
            truncation_cutoff=None,
            update_emas=False,
        ):
        bs,n,dim = zs.shape
        zs_new = zs.reshape(bs*n,dim)
        c_new = c[:,None,:].repeat(1,n,1).reshape(bs*n,-1)
        if 'resnet_feats' in cond:
            cond_new = {**cond}
            cond_new['resnet_feats'] = cond['resnet_feats'][:,None,:].repeat(1,n,1).reshape(bs*n,-1)
        else:
            cond_new = cond
        # print(zs.shape)
        # print(c.shape, c.dtype)
        # print(zs_new.shape)
        # print(c_new.shape)
        ans = self.mapping(zs_new, c_new, cond_new, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        ans = ans.view(bs,n,n,dim).permute(1,2,0,3)[range(n),range(n)].permute(1,0,2)
        return ans

    def synthesis(
            self, ws, c, cond,
            neural_rendering_resolution=None,
            update_emas=False,
            cache_backbone=False,
            use_cached_backbone=False,
            latent_injection=None,
            stop_level=None,
            force_rays=None,
            triplane_crop=None,
            cull_clouds=None,
            binarize_clouds=None,
            normalize_images=True,
            return_more=False,
            **synthesis_kwargs,
        ):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        if force_rays is None:
            ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)
            # for k in ['ray_origins', 'ray_directions']:
            #     uutil.pdump(eval(k), f'/dev/shm/{k}.pkl')
        elif isinstance(force_rays, dict):
            # bw = self.rendering_kwargs['box_warp']
            bs = len(ws)
            r = neural_rendering_resolution
            ray_origins = force_rays['ray_origins']
            ray_directions = force_rays['ray_directions']
            assert ray_origins.shape==ray_directions.shape==(bs,3,r,r)
            ray_origins = utorch.einops.rearrange(ray_origins, 'bs ch h w -> bs (h w) ch')
            ray_directions = utorch.einops.rearrange(ray_directions, 'bs ch h w -> bs (h w) ch')
        else:
            assert False, f'force_rays not understood'

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            if not return_more:
                planes = self.backbone.synthesis(ws, cond, update_emas=update_emas, latent_injection=latent_injection, stop_level=stop_level, **synthesis_kwargs)
            else:
                planes,locs_planes = self.backbone.synthesis(ws, cond, update_emas=update_emas, latent_injection=latent_injection, stop_level=stop_level, return_more=True, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Reshape output into three 32-channel planes
        # planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        planes = planes.view(
            len(planes),
            3,
            self.triplane_width*self.rendering_kwargs.get('triplane_depth',1),
            planes.shape[-2],
            planes.shape[-1],
        )  # changed for yichun's multiplane (shu added)

        # Perform volume rendering
        feature_samples, depth_samples, weights_samples, xyz_samples = self.renderer(
            planes, self.decoder,
            ray_origins, ray_directions,
            self.rendering_kwargs,
            triplane_crop=triplane_crop,
            cull_clouds=cull_clouds,
            binarize_clouds=binarize_clouds,
        ) # channels last
        # print(feature_samples.shape)
        # print(depth_samples.shape)
        # print(weights_samples.shape)

        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        xyz_image = xyz_samples.permute(0, 2, 1).reshape(N, xyz_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
        weights_image = weights_samples.permute(0, 2, 1).reshape(N, 1, H, W)
        # print(feature_image.shape)
        # print(depth_image.shape)
        xyz_image = 0.5 * (xyz_image + 1) * torch.tensor([-1,1,-1]).to(xyz_image.device)[None,:,None,None]

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]
        sr_image = self.superresolution(rgb_image, feature_image, ws, noise_mode=self.rendering_kwargs['superresolution_noise_mode'], **{k:synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != 'noise_mode'})

        # ans = {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image}
        ans = {
            'image': sr_image,
            'image_raw': rgb_image,
            'image_depth': depth_image,
            'triplane': planes,
            'image_weights': weights_image,
            'image_xyz': xyz_image,
        }
        if self.rendering_kwargs.get('tanh_rgb_output', False):
            ans['image'] = torch.tanh(ans['image'])
            ans['image_raw'] = torch.tanh(ans['image_raw'])
        if not normalize_images:
            ans['image'] = 0.5*ans['image'] + 0.5
            ans['image_raw'] = 0.5*ans['image_raw'] + 0.5
        if return_more:
            ans['locals'] = locals()
        return ans
    
    def sample(
            self, coordinates, directions, z, c, cond,
            truncation_psi=1,
            truncation_cutoff=None,
            update_emas=False,
            **synthesis_kwargs,
        ):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes. 
        ws = self.mapping(z, c, cond, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, cond, update_emas=update_emas, **synthesis_kwargs)
        # planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        planes = planes.view(
            len(planes),
            3,
            self.triplane_width*self.rendering_kwargs.get('triplane_depth',1),
            planes.shape[-2],
            planes.shape[-1],
        )  # changed for yichun's multiplane (shu added)        return self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)

    def sample_mixed(
            self, coordinates, directions, ws, cond,
            truncation_psi=1,
            truncation_cutoff=None,
            update_emas=False,
            # triplane_crop=None,
            # cull_clouds=None,
            **synthesis_kwargs,
        ):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws, cond, update_emas = update_emas, **synthesis_kwargs)
        # planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        planes = planes.view(
            len(planes),
            3,
            self.triplane_width*self.rendering_kwargs.get('triplane_depth',1),
            planes.shape[-2],
            planes.shape[-1],
        )  # changed for yichun's multiplane (shu added)
        return self.renderer.run_model(
            planes, self.decoder,
            coordinates, directions,
            self.rendering_kwargs,
            # triplane_crop=triplane_crop,
            # cull_clouds=cull_clouds,
        )

    def forward(
            self, z, c, cond,
            truncation_psi=1,
            truncation_cutoff=None,
            neural_rendering_resolution=None,
            update_emas=False,
            cache_backbone=False,
            use_cached_backbone=False,
            **synthesis_kwargs,
        ):
        # Render a batch of generated images.
        ws = self.mapping(z, c, cond, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c, cond, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)
    def f(
            self,
            x,
            truncation_psi=1,
            truncation_cutoff=None,
            latent_injection=None,
            force_rays=None,
            stop_level=None,
            normalize_images=False,
            return_more=False,
        ):
        # get device and bs
        if 'ws' in x:
            bs,device = len(x['ws']), x['ws'].device
        elif 'camera_params' in x:
            bs,device = len(x['camera_params']), x['camera_params'].device
        # elif 'cond' in x:
        #     bs,device = len(x['cond']), x['cond'].device
        elif 'elevations' in x:
            bs,device = len(x['elevations']), x['elevations'].device
        elif 'zs' in x:
            bs,device = len(x['zs']), x['zs'].device
        elif 'z' in x:
            bs,device = len(x['z']), x['z'].device
        else:
            device = self.backbone.mapping.embed.weight.device
        dtype = self.backbone.mapping.embed.weight.dtype

        # convenience merge
        if 'latent_injection' in x:
            if latent_injection==None:
                latent_injection = x['latent_injection']
            else:
                latent_injection = uutil.Dict({**latent_injection})
                latent_injection.update(x['latent_injection'])

        # latents
        if 'zs' not in x and 'ws' not in x:
            if 'z' not in x:
                x['z'] = torch.tensor(np.stack([
                    np.random.RandomState(s).randn(self.z_dim)
                    for s in x['seeds']
                ]), device=device, dtype=dtype)
            x['zs'] = x['z'][:,None,:].expand(-1,self.backbone.num_ws,-1)

        # output cameras
        if 'camera_params' not in x:
            if 'distances' not in x:
                x['distances'] = torch.ones_like(x['elevations'])
            if 'fovs' not in x:
                x['fovs'] = 30 * torch.ones_like(x['elevations'])
            # camera_params = []
            # for elev,azim,dist,fov in zip(x['elevations'], x['azimuths'], x['distances'], x['fovs'].cpu().numpy()):
            #     angle_y,angle_p = azim/180*np.pi, -elev/180*np.pi
            #     cam2world_pose = LookAtPoseSampler.sample(
            #         np.pi/2 + angle_y,
            #         np.pi/2 + angle_p,
            #         torch.zeros(3, device=angle_y.device),
            #         radius=dist,
            #         device=angle_y.device,
            #     )
            #     fl = 0.5/np.tan(fov*np.pi/180)
            #     intrinsics = torch.tensor([
            #         [fl, 0, 0.5,],
            #         [0, fl, 0.5,],
            #         [0,  0, 1.0,],
            #     ], device=angle_y.device)
            #     camera_params.append(torch.cat([cam2world_pose.flatten(), intrinsics.flatten()]))
            # x['camera_params'] = camera_params = torch.stack(camera_params).to(dtype)
            x['camera_params'] = torch.stack([
                dklustr.camera_params_to_matrix('eg3d_lustrousB', elev=elev, azim=azim, dist=dist, fov=fov)['camera_label']
                for elev,azim,dist,fov in zip(x['elevations'], x['azimuths'], x['distances'], x['fovs'])
            ]).to(dtype).to(device)

        # force rays by default (calculate them here)
        force_rays = (x['force_rays'] if 'force_rays' in x else None) or force_rays
        neural_rendering_resolution = x['neural_rendering_resolution'] \
            if 'neural_rendering_resolution' in x else self.neural_rendering_resolution
        if force_rays==None:
            # regular perspective
            cam2world_matrix = x['camera_params'][:, :16].view(-1, 4, 4)
            intrinsics = x['camera_params'][:, 16:25].view(-1, 3, 3)
            res = neural_rendering_resolution
            ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, res)
            ray_origins = utorch.einops.rearrange(ray_origins, 'bs (h w) ch -> bs ch h w', h=res, w=res)
            ray_directions = utorch.einops.rearrange(ray_directions, 'bs (h w) ch -> bs ch h w', h=res, w=res)

            # detect + replace orthographic ones (negative fov)
            for i,intr in enumerate(intrinsics):
                if intr[0,0]<0:
                    r = dklustr.get_rays_ortho(
                        x['elevations'][i],
                        x['azimuths'][i],
                        x['distances'][i],
                        self.rendering_kwargs['box_warp'],
                        res,
                        device=device,
                    )
                    ray_origins[i] = r['ray_origins']
                    ray_directions[i] = r['ray_directions']

            # force
            x['force_rays'] = force_rays = {
                'ray_origins': ray_origins,
                'ray_directions': ray_directions,
            }

        # condition cameras
        x['conditioning_params'] = x['camera_params']
        # if 'conditioning_params' not in x and 'ws' not in x:
        #     x['conditioning_params'] = 'kongo'
        # if isinstance(x['conditioning_params'], str):
        #     if x['conditioning_params']=='kongo':
        #         elev,azim,dist,fov = 0,0,1,12
        #         fl = 0.5/np.tan(fov*np.pi/180)
        #     elif x['conditioning_params']=='ffhq':
        #         elev,azim,dist = 0,0,2.7
        #         fl = 4.2647
        #     else:
        #         assert 0
        #     angle_y,angle_p = azim/180*np.pi, elev/180*np.pi
        #     cam2world_pose = LookAtPoseSampler.sample(
        #         np.pi/2 + angle_y,
        #         np.pi/2 + angle_p,
        #         torch.zeros(3, device=device),
        #         radius=dist,
        #         device=device,
        #     )
        #     intrinsics = torch.tensor([
        #         [fl, 0, 0.5,],
        #         [0, fl, 0.5,],
        #         [0,  0, 1.0,],
        #     ], device=device)
        #     cond = torch.cat([cam2world_pose.flatten(), intrinsics.flatten()])
        #     x['conditioning_params'] = cond[None,].expand(bs,25)
        # elif x['conditioning_params'] is None:
        #     x['conditioning_params'] = x['camera_params']
        # # x['conditioning_params'] = x['conditioning_params'].to(dtype)

        # ws mapping
        if 'ws' not in x:
            # print(x['zs'].dtype, x['zs'].device)
            # print(x['conditioning_params'].dtype, x['conditioning_params'].device)
            ws = self.mapping_zplus(
                x['zs'], x['conditioning_params'], x['cond'],
                truncation_psi=truncation_psi,
                truncation_cutoff=truncation_cutoff,
            )
            x['ws'] = ws#.to(dtype)

        # synthesis
        _ws = x['ws']
        if latent_injection!=None:
            if 'dw' in latent_injection:
                _ws = _ws + latent_injection['dw']
            if 'dws' in latent_injection:
                _ws = _ws + latent_injection['dws']
        normalize_images = x['normalize_images'] if 'normalize_images' in x else normalize_images
        synth = self.synthesis(
            _ws, x['camera_params'], x['cond'],
            latent_injection=latent_injection,
            triplane_crop=x['triplane_crop'] if 'triplane_crop' in x else None,
            cull_clouds=x['cull_clouds'] if 'cull_clouds' in x else None,
            binarize_clouds=x['binarize_clouds'] if 'binarize_clouds' in x else None,
            force_rays=force_rays,
            stop_level=stop_level,
            normalize_images=normalize_images,
            neural_rendering_resolution=neural_rendering_resolution,
            update_emas=x['update_emas'] if 'update_emas' in x else False,
            return_more=return_more,
        )
        ret = uutil.Dict({
            # 'image': 0.5*synth['image'] + 0.5,
            # 'image_raw': 0.5*synth['image_raw'] + 0.5,
            'image': synth['image'],
            'image_raw': synth['image_raw'],
            'image_depth': synth['image_depth'],
            'image_weights': synth['image_weights'],
            'triplane': synth['triplane'],
            'image_xyz': synth['image_xyz'],
            'normalize_images': normalize_images,
        })
        x.update(ret)

        # image pasting
        if 'paste_params' in x and x['paste_params']!=None:
            ret['image_prepaste'] = ret['image']
            paste = paste_front(self, x, ret, **x['paste_params'])
            ret['paste'] = paste
            ret['image'] = paste['image']

        if return_more:
            ret.update({
                'locals': locals(),
            })
        return ret

    def set_force_sigmoid(self, state):
        return self.decoder.set_force_sigmoid(state)


from training.networks_stylegan2 import FullyConnectedLayer

class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64
        self.force_sigmoid = False

        self.net = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options['decoder_lr_mul']),
            torch.nn.Softplus(),
            FullyConnectedLayer(self.hidden_dim, 1 + options['decoder_output_dim'], lr_multiplier=options['decoder_lr_mul'])
        )
        
    def forward(self, sampled_features, ray_directions, force_sigmoid=None):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features
        force_sigmoid = force_sigmoid or self.force_sigmoid

        N, M, C = x.shape
        x = x.view(N*M, C)

        x = self.net(x)
        x = x.view(N, M, -1)
        if force_sigmoid:
            rgb = torch.sigmoid(x[...,1:])
        else:
            rgb = torch.sigmoid(x[..., 1:])*(1 + 2*0.001) - 0.001 # Uses sigmoid clamping from MipNeRF
        sigma = x[..., 0:1]
        return {'rgb': rgb, 'sigma': sigma}

    def set_force_sigmoid(self, state):
        self.force_sigmoid = state
        return self.force_sigmoid




######## pasting utils ########

def sample_orthofront(front_rgb, view_xyz, bw):
    frgb,vxyz = front_rgb, view_xyz
    vij = 1 - (vxyz[:,[1,0]]+bw/2)/bw
    ans = nn.functional.grid_sample(
        frgb.permute(0,1,3,2),
        vij.permute(0,2,3,1)*2-1,
        padding_mode='border',
        mode='bilinear',
    )
    return ans
def get_front_occlusion(G, x, out, offset=0.01):
    ro = out['image_xyz'] #.detach().clone()
    ro = ro * torch.tensor([-1,1,-1], device=ro.device)[None,:,None,None]
    ro[:,2,:,:] -= G.rendering_kwargs['ray_start'] - offset
    rd = torch.zeros_like(out['image_xyz'])
    rd[:,2,:,:] = 1
    xin = {**x}
    xin['paste_params'] = None
    xin['force_rays'] = {
        'ray_origins': ro,
        'ray_directions': rd,
    }
    ans = G.f(xin, return_more=True)['image_weights']
    return ans
def get_front_weights(G, x,):
    # ro = out['image_xyz'] #.detach().clone()
    # ro = ro * torch.tensor([-1,1,-1], device=ro.device)[None,:,None,None]
    # ro[:,2,:,:] -= G.rendering_kwargs['ray_start'] - offset
    # rd = torch.zeros_like(out['image_xyz'])
    # rd[:,2,:,:] = 1
    device = x['cond']['image_ortho_front'].device
    xin = {
        k: v for k,v in x.items()
        if k not in ['paste_params', 'camera_params', 'conditioning_params', 'force_rays']
    }
    # xin['paste_params'] = None
    # xin['force_rays'] = {
    #     'ray_origins': ro,
    #     'ray_directions': rd,
    # }
    xin['elevations'] = torch.zeros(1).to(device)
    xin['azimuths'] = torch.zeros(1).to(device)
    xin['fovs'] = -torch.ones(1).to(device)
    ans = G.f(xin, return_more=True)['image_weights']
    return ans
def get_xyz_discrepancy(xyz, rays):
    a = rays['ray_origins']
    n = rays['ray_directions']
    p = xyz * torch.tensor([-1,1,-1], device=xyz.device)[None,:,None,None]
    ans = ( (p-a) - ((p-a)*n).sum(dim=1,keepdims=True) * n ).norm(2, dim=1, keepdim=True)
    return ans

def paste_front(
        G, x, out, mode='default',
        thresh_weight=0.95,
        thresh_edges=0.02,
        thresh_occ=0.05, offset_occ=0.01,
        thresh_dxyz=0.01,
        front_weight_erosion=0,
        grad_sample=False,
        force_image=None,
        **kwargs,
        ):
    view_xyz = out['image_xyz']
    view_alpha = out['image_weights']
    front_rgb = x['cond']['image_ortho_front']
    
    # masks
    with torch.no_grad():
        # mask visible weights
        wmask = (nn.functional.interpolate(
            out['image_weights'],
            front_rgb.shape[-1],
            mode='bilinear',
        )>thresh_weight).float()
        
        # mask deep crevaces
        smask = kornia.filters.sobel(nn.functional.interpolate(
            out['image_xyz'],
            front_rgb.shape[-1],
            mode='bilinear',
        )).norm(2,dim=1,keepdim=True)
        smask = (smask<thresh_edges).float()
        
        # mask occlusion from front
        fmask = (get_front_occlusion(G, x, out, offset=offset_occ)<thresh_occ).float()
        fmask = nn.functional.interpolate(fmask, front_rgb.shape[-1], mode='bilinear')

        # mask xyz discrepancies
        dmask = get_xyz_discrepancy(out['image_xyz'], x['force_rays'])
        dmask = nn.functional.interpolate(dmask, front_rgb.shape[-1], mode='nearest')
        dmask = (dmask<thresh_dxyz).float()

        # mask edges from front
        if front_weight_erosion>=1:
            frontw = get_front_weights(G, x)
            e = front_weight_erosion
            fwmask = kornia.morphology.erosion(
                (frontw>0.5).float(),
                torch.ones(e,e).to(out['image_weights'].device),
            )
            fwmask = sample_orthofront(
                fwmask,
                nn.functional.interpolate(view_xyz, front_rgb.shape[-1], mode='bilinear'),
                G.rendering_kwargs['box_warp'],
            )
            fwmask = nn.functional.interpolate(fwmask, front_rgb.shape[-1], mode='nearest')
        else:
            frontw = None
            fwmask = torch.ones(*dmask.shape).to(dmask.device)

        # composite
        mask = wmask*smask*fmask*dmask*fwmask

    # generate paste
    if force_image is None:
        tocopy = front_rgb if not x['normalize_images'] else front_rgb*2-1
    else:
        tocopy = force_image.t()[None,].to(mask.device)
    with (uutil.contextlib.nullcontext() if grad_sample else torch.no_grad()):
        paste = sample_orthofront(
            tocopy,
            nn.functional.interpolate(view_xyz, front_rgb.shape[-1], mode='bilinear'),
            G.rendering_kwargs['box_warp'],
        )
    ans = torch.lerp(out['image'], paste, mask)
    return uutil.Dict({
        'image': ans,
        'paste': paste,
        'mask': mask,
        'mask_weights': wmask,
        'mask_edges': smask,
        'mask_occ': fmask,
        'mask_dxyz': dmask,
        'mask_frontweight': fwmask,
        'frontweight': frontw,
    })



