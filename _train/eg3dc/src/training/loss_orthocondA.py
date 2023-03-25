# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing
from torch import nn
import kornia

import _util.util_v1 as uutil
import _util.pytorch_v1 as utorch
import _util.twodee_v1 as u2d


#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

def mask_view_orthofront(front_xyz, front_alpha, view_xyz, view_alpha, boxwarp):
    fxyz,falpha,vxyz,valpha,bw = front_xyz, front_alpha, view_xyz, view_alpha, boxwarp
    fz = fxyz[:,2:3]
    vij = 1 - (vxyz[:,[1,0]]+bw/2)/bw
    vz = vxyz[:,2:3]
    qs = nn.functional.grid_sample(
        torch.cat([
            (falpha>0.5).float(),
            # x_rgb_f.image.t()[None].to(device),
            fz,
        ], dim=1).permute(0,1,3,2),
        vij.permute(0,2,3,1)*2-1,
        padding_mode='zeros',
        mode='nearest',
    )
    zmask = (vz - qs[:,-1:]) < (1.5/255) * bw
    ans = qs[:,:-1] * zmask * (valpha > 0.5)
    # I(qs[:,-1] * zmask).norm().convert('RGBA').bg().right(x_rgb_f.image)
    # I(ans).convert('RGBA').bg().right(x_rgb_f.image)
    return ans

class StyleGAN2LossOrthoCondA(Loss):
    def __init__(
            self, device, G, D, lpips_model,
            augment_pipe=None, r1_gamma=10, style_mixing_prob=0,
            pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False,
            blur_init_sigma=0, blur_fade_kimg=0,
            r1_gamma_init=0, r1_gamma_fade_kimg=0,
            neural_rendering_resolution_initial=64,
            neural_rendering_resolution_final=None,
            neural_rendering_resolution_fade_kimg=0,
            gpc_reg_fade_kimg=1000, gpc_reg_prob=None,
            dual_discrimination=False, filter_mode='antialiased',
            lambda_Gcond_lpips=10.0, lambda_Gcond_l1=1.0,
            lambda_Gcond_alpha_l2=0.0, lambda_Gcond_depth_l2=0.0,
            lambda_Gcond_sides_lpips=0.0, lambda_Gcond_sides_l1=0.0,
            lambda_Gcond_sides_alpha_l2=0.0, lambda_Gcond_sides_depth_l2=0.0,
            lambda_Gcond_back_lpips=0.0, lambda_Gcond_back_l1=0.0,
            lambda_Gcond_back_alpha_l2=0.0, lambda_Gcond_back_depth_l2=0.0,
            lambda_Gcond_rand_lpips=0.0, lambda_Gcond_rand_l1=0.0,
            lambda_Gcond_rand_alpha_l2=0.0, lambda_Gcond_rand_depth_l2=0.0,
            lossmask_mode_adv='none', lossmask_mode_recon='none',
            lambda_recon_lpips=0.0, lambda_recon_l1=0.0,
            lambda_recon_alpha_l2=0.0, lambda_recon_depth_l2=0.0,
            paste_params_mode=None,
            ):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.r1_gamma_init      = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
        self.blur_raw_target = True
        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)

        self.lpips_model = lpips_model
        self.lambda_Gcond_lpips = lambda_Gcond_lpips
        self.lambda_Gcond_l1 = lambda_Gcond_l1
        self.lambda_Gcond_alpha_l2 = lambda_Gcond_alpha_l2
        self.lambda_Gcond_depth_l2 = lambda_Gcond_depth_l2
        self.lambda_Gcond_sides_lpips = lambda_Gcond_sides_lpips
        self.lambda_Gcond_sides_l1 = lambda_Gcond_sides_l1
        self.lambda_Gcond_sides_alpha_l2 = lambda_Gcond_sides_alpha_l2
        self.lambda_Gcond_sides_depth_l2 = lambda_Gcond_sides_depth_l2
        self.lambda_Gcond_back_lpips = lambda_Gcond_back_lpips
        self.lambda_Gcond_back_l1 = lambda_Gcond_back_l1
        self.lambda_Gcond_back_alpha_l2 = lambda_Gcond_back_alpha_l2
        self.lambda_Gcond_back_depth_l2 = lambda_Gcond_back_depth_l2
        self.lambda_Gcond_rand_lpips = lambda_Gcond_rand_lpips
        self.lambda_Gcond_rand_l1 = lambda_Gcond_rand_l1
        self.lambda_Gcond_rand_alpha_l2 = lambda_Gcond_rand_alpha_l2
        self.lambda_Gcond_rand_depth_l2 = lambda_Gcond_rand_depth_l2

        self.lossmask_mode_adv = lossmask_mode_adv
        self.lossmask_mode_recon = lossmask_mode_recon
        self.lambda_recon_lpips = lambda_recon_lpips
        self.lambda_recon_l1 = lambda_recon_l1
        self.lambda_recon_alpha_l2 = lambda_recon_alpha_l2
        self.lambda_recon_depth_l2 = lambda_recon_depth_l2

        default_pp = {
            'mode': 'default',
            'thresh_weight': 0.95,
            'thresh_edges': 0.02,
            'thresh_occ': 0.05, 'offset_occ': 0.01,
            'thresh_dxyz': 0.000005,
        }
        self.paste_params_mode = paste_params_mode
        if self.paste_params_mode=='A':
            self.paste_params = {
                **default_pp,
                'grad_sample': False,
            }
        elif self.paste_params_mode=='Agrad':
            self.paste_params = {
                **default_pp,
                'grad_sample': True,
            }
        elif self.paste_params_mode in [None, 'none']:
            self.paste_params = None
        else:
            assert 0
        return

    def run_G(self, z, c, cond, swapping_prob, neural_rendering_resolution, update_emas=False):
        if swapping_prob is not None:
            c_swapped = torch.roll(c.clone(), 1, 0)
            c_gen_conditioning = torch.where(torch.rand((c.shape[0], 1), device=c.device) < swapping_prob, c_swapped, c)
        else:
            c_gen_conditioning = torch.zeros_like(c)

        ws = self.G.mapping(z, c_gen_conditioning, cond, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, cond, update_emas=False)[:, cutoff:]
        # gen_output = self.G.synthesis(ws, c, cond, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas)
        gen_output = self.G.f({
            'ws': ws,
            'camera_params': c,
            'cond': cond,
            'normalize_images': True,
            'neural_rendering_resolution': neural_rendering_resolution,
            'update_emas': update_emas,
            'paste_params': self.paste_params,
        })
        return gen_output, ws

    def run_D(self, img, c, cond, blur_sigma=0, blur_sigma_raw=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        if self.augment_pipe is not None:
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                                    torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear', antialias=True)],
                                                    dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img['image'].shape[1]:], size=img['image_raw'].shape[2:], mode='bilinear', antialias=True)

        logits = self.D(img, c, cond, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, real_cond, gen_z, gen_c, gain, cur_nimg):
        # assert phase in [
        #     'Gmain', 'Gcond', 'Gside-left', 'Gside-right', 'Grand', 'Greg', 'Gboth',
        #     'Dmain', 'Dcond', 'Dside-left', 'Dside-right', 'Drand', 'Dreg', 'Dboth',
        # ]
        if self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        if self.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

        real_img = {
            'image': real_img,
            'image_raw': real_img_raw,
            'image_raw_noblur': torch.nn.functional.interpolate(real_img, real_img_raw.shape[-1], mode='bilinear'),
        }


        ##### calc loss mask #####

        lmask = mask_view_orthofront(
            real_cond['image_ortho_front_xyz'],
            real_cond['image_ortho_front_alpha'],
            real_cond['image_xyz'],
            real_cond['image_alpha'],
            self.G.rendering_kwargs['box_warp'],
        )
        if self.lossmask_mode_adv!='none':
            k = int(self.lossmask_mode_adv.split('_')[-1])
            lmask_adv = 1 - kornia.morphology.erosion(lmask, torch.ones(k,k, device=lmask.device))
            lmask_adv_raw = (torch.nn.functional.interpolate(
                lmask_adv, real_img['image_raw'].shape[-1], mode='bilinear',
            )>0.5).float()
        if self.lossmask_mode_recon!='none':
            k = int(self.lossmask_mode_recon.split('_')[-1])
            lmask_recon = kornia.morphology.dilation(lmask, torch.ones(k,k, device=lmask.device))
            lmask_recon_raw = (torch.nn.functional.interpolate(
                lmask_recon, real_img['image_raw'].shape[-1], mode='bilinear',
            )>0.5).float()


        ################ ADDED PHASES ################

        # conditioned synthesis must match condition
        if phase in ['Gcond', 'Gboth']:
            if self.G.cond_mode=='none':
                pass
            elif self.G.cond_mode.startswith('ortho_front.'):
                cm = set(self.G.cond_mode.split('.'))

                # calc front view loss
                if True:
                    # render front ortho views
                    xin = {
                        'z': gen_z,
                        'cond': real_cond,
                        'camera_params': real_cond['image_ortho_front_camera'],
                        'elevations': torch.zeros(len(gen_z), device=gen_z.device),
                        'azimuths': torch.zeros(len(gen_z), device=gen_z.device),
                        'distances': torch.ones(len(gen_z), device=gen_z.device),  # forgive me for I have sinned
                        'paste_params': self.paste_params,
                    }
                    out = self.G.f(xin)

                    # calc rgb losses
                    loss_Gcond_lpips = self.lpips_model(out['image'], real_cond['image_ortho_front']).mean()
                    loss_Gcond_l1 = (out['image'] - real_cond['image_ortho_front']).abs().mean()
                    training_stats.report('Loss/G/cond/lpips', loss_Gcond_lpips)
                    training_stats.report('Loss/G/cond/l1', loss_Gcond_l1)

                    # calc alpha loss
                    s = out['image_weights'].shape[-1]
                    k = 2
                    gt_alpha = real_cond['image_ortho_front_alpha']
                    gt_alpha = torch.nn.functional.interpolate(gt_alpha, s, mode='bilinear')
                    msk = torch.nn.functional.conv2d(
                        gt_alpha,
                        torch.ones(1,1, 2*k+1,2*k+1).to(gt_alpha.device),
                        stride=1,
                        padding=k,
                    ) / (2*k+1)**2
                    msk = (msk-0.5).abs()*2 > 0.5
                    loss_Gcond_alpha_l2 = (out['image_weights']-gt_alpha).pow(2).mul(msk.float()).mean()
                    training_stats.report('Loss/G/cond/alpha_l2', loss_Gcond_alpha_l2)

                    # calc depthz loss
                    gt_xyz = real_cond['image_ortho_front_xyz']
                    gt_xyz = torch.nn.functional.interpolate(gt_xyz, s, mode='bilinear')
                    with torch.no_grad():
                        mskz = msk & (out['image_weights'] > 0.5) & (gt_alpha>0.5)
                    loss_Gcond_depth_l2 = (out['image_xyz'][:,2] - gt_xyz[:,2]).pow(2).mul(mskz.float()).mean()
                    training_stats.report('Loss/G/cond/depth_l2', loss_Gcond_depth_l2)

                # aggregate
                loss_Gcond = (
                    self.lambda_Gcond_lpips * loss_Gcond_lpips +
                    self.lambda_Gcond_l1 * loss_Gcond_l1 +
                    self.lambda_Gcond_alpha_l2 * loss_Gcond_alpha_l2 +
                    self.lambda_Gcond_depth_l2 * loss_Gcond_depth_l2
                )
                training_stats.report('Loss/G/cond', loss_Gcond)
                loss_Gcond.backward()
            else:
                assert 0, f'cond_mode {self.G.cond_mode} not understood'

        # conditioned synthesis must match left/right/back sides
        if phase in ['Gside-left', 'Gside-right', 'Gside-back']:
            if self.G.cond_mode=='none':
                pass
            elif self.G.cond_mode.startswith('ortho_front.'):
                cm = set(self.G.cond_mode.split('.'))

                # calc side losses
                view = phase.split('-')[-1]

                # render front ortho views
                az = {'left': 90, 'right': -90, 'back': 180}[view]
                xin = {
                    'z': gen_z,
                    'cond': real_cond,
                    'camera_params': real_cond[f'image_ortho_{view}_camera'],
                    'elevations': torch.zeros(len(gen_z), device=gen_z.device),
                    'azimuths': az * torch.ones(len(gen_z), device=gen_z.device),
                    'distances': torch.ones(len(gen_z), device=gen_z.device),  # forgive me for I have sinned
                    'paste_params': self.paste_params,
                }
                out = self.G.f(xin)

                # calc rgb losses
                loss_Gcond_sides_lpips = self.lpips_model(out['image'], real_cond[f'image_ortho_{view}']).mean()
                loss_Gcond_sides_l1 = (out['image'] - real_cond[f'image_ortho_{view}']).abs().mean()
                # training_stats.report('Loss/G/cond/lpips', loss_Gcond_lpips)
                # training_stats.report('Loss/G/cond/l1', loss_Gcond_l1)

                # calc alpha loss
                s = out['image_weights'].shape[-1]
                k = 2
                gt_alpha = real_cond[f'image_ortho_{view}_alpha']
                gt_alpha = torch.nn.functional.interpolate(gt_alpha, s, mode='bilinear')
                msk = torch.nn.functional.conv2d(
                    gt_alpha,
                    torch.ones(1,1, 2*k+1,2*k+1).to(gt_alpha.device),
                    stride=1,
                    padding=k,
                ) / (2*k+1)**2
                msk = (msk-0.5).abs()*2 > 0.5
                loss_Gcond_sides_alpha_l2 = (out['image_weights']-gt_alpha).pow(2).mul(msk.float()).mean()
                # training_stats.report('Loss/G/cond/alpha_l2', loss_Gcond_alpha_l2)

                # calc depthz loss
                gt_xyz = real_cond[f'image_ortho_{view}_xyz']
                gt_xyz = torch.nn.functional.interpolate(gt_xyz, s, mode='bilinear')
                with torch.no_grad():
                    mskz = msk & (out['image_weights'] > 0.5) & (gt_alpha>0.5)
                if view=='back':
                    loss_Gcond_sides_depth_l2 = (out['image_xyz'][:,2] - gt_xyz[:,2]).pow(2).mul(mskz.float()).mean()
                else:
                    loss_Gcond_sides_depth_l2 = (out['image_xyz'][:,0] - gt_xyz[:,0]).pow(2).mul(mskz.float()).mean()
                # training_stats.report('Loss/G/cond/depth_l2', loss_Gcond_depth_l2)
                # loss_Gcond_sides_lpips = loss_Gcond_sides_lpips / 2
                # loss_Gcond_sides_l1 = loss_Gcond_sides_l1 / 2
                # loss_Gcond_sides_alpha_l2 = loss_Gcond_sides_alpha_l2 / 2
                # loss_Gcond_sides_depth_l2 = loss_Gcond_sides_depth_l2 / 2
                training_stats.report(f'Loss/G/sides/{view}/lpips', loss_Gcond_sides_lpips)
                training_stats.report(f'Loss/G/sides/{view}/l1', loss_Gcond_sides_l1)
                training_stats.report(f'Loss/G/sides/{view}/alpha_l2', loss_Gcond_sides_alpha_l2)
                training_stats.report(f'Loss/G/sides/{view}/depth_l2', loss_Gcond_sides_depth_l2)

                # aggregate
                if view in ['left', 'right']:
                    loss_Gcond = (
                        self.lambda_Gcond_sides_lpips * loss_Gcond_sides_lpips +
                        self.lambda_Gcond_sides_l1 * loss_Gcond_sides_l1 +
                        self.lambda_Gcond_sides_alpha_l2 * loss_Gcond_sides_alpha_l2 +
                        self.lambda_Gcond_sides_depth_l2 * loss_Gcond_sides_depth_l2
                    )
                elif view=='back':
                    loss_Gcond = (
                        self.lambda_Gcond_back_lpips * loss_Gcond_sides_lpips +
                        self.lambda_Gcond_back_l1 * loss_Gcond_sides_l1 +
                        self.lambda_Gcond_back_alpha_l2 * loss_Gcond_sides_alpha_l2 +
                        self.lambda_Gcond_back_depth_l2 * loss_Gcond_sides_depth_l2
                    )
                else:
                    assert 0
                training_stats.report(f'Loss/G/sides/{view}', loss_Gcond)
                loss_Gcond.backward()
            else:
                assert 0, f'cond_mode {self.G.cond_mode} not understood'

        # conditioned synthesis must match random view
        if phase in ['Grand', 'Gboth']:
            if self.G.cond_mode=='none':
                pass
            elif self.G.cond_mode.startswith('ortho_front.'):
                cm = set(self.G.cond_mode.split('.'))

                # calc rand view loss
                if True:
                    # render rand ortho views
                    xin = {
                        'z': gen_z,
                        'cond': real_cond,
                        'camera_params': real_cond['image_camera'],
                        # 'elevations': torch.zeros(len(gen_z), device=gen_z.device),
                        # 'azimuths': torch.zeros(len(gen_z), device=gen_z.device),
                        # 'distances': torch.ones(len(gen_z), device=gen_z.device),  # forgive me for I have sinned
                        'paste_params': self.paste_params,
                    }
                    out = self.G.f(xin)

                    # calc rgb losses
                    loss_Gcond_rand_lpips = self.lpips_model(out['image'], real_cond['image']).mean()
                    loss_Gcond_rand_l1 = (out['image'] - real_cond['image']).abs().mean()
                    training_stats.report('Loss/G/rand/lpips', loss_Gcond_rand_lpips)
                    training_stats.report('Loss/G/rand/l1', loss_Gcond_rand_l1)

                    # calc alpha loss
                    s = out['image_weights'].shape[-1]
                    k = 2
                    gt_alpha = real_cond['image_alpha']
                    gt_alpha = torch.nn.functional.interpolate(gt_alpha, s, mode='bilinear')
                    msk = torch.nn.functional.conv2d(
                        gt_alpha,
                        torch.ones(1,1, 2*k+1,2*k+1).to(gt_alpha.device),
                        stride=1,
                        padding=k,
                    ) / (2*k+1)**2
                    msk = (msk-0.5).abs()*2 > 0.5
                    loss_Gcond_rand_alpha_l2 = (out['image_weights']-gt_alpha).pow(2).mul(msk.float()).mean()
                    training_stats.report('Loss/G/rand/alpha_l2', loss_Gcond_rand_alpha_l2)

                    # calc depthz loss
                    gt_xyz = real_cond['image_xyz']
                    gt_xyz = torch.nn.functional.interpolate(gt_xyz, s, mode='bilinear')
                    with torch.no_grad():
                        mskz = msk & (out['image_weights'] > 0.5) & (gt_alpha>0.5)
                    loss_Gcond_rand_depth_l2 = (out['image_xyz'] - gt_xyz).pow(2).sum(dim=1).sqrt().mul(mskz.float()).mean()
                    training_stats.report('Loss/G/rand/depth_l2', loss_Gcond_rand_depth_l2)

                # aggregate
                loss_Gcond = (
                    self.lambda_Gcond_rand_lpips * loss_Gcond_rand_lpips +
                    self.lambda_Gcond_rand_l1 * loss_Gcond_rand_l1 +
                    self.lambda_Gcond_rand_alpha_l2 * loss_Gcond_rand_alpha_l2 +
                    self.lambda_Gcond_rand_depth_l2 * loss_Gcond_rand_depth_l2
                )
                training_stats.report('Loss/G/rand', loss_Gcond)
                loss_Gcond.backward()
            else:
                assert 0, f'cond_mode {self.G.cond_mode} not understood'

        if phase in ['Dcond', 'Dboth']:
            pass
        if phase in ['Dsides', 'Dboth']:
            pass
        if phase in ['Drand', 'Dboth']:
            pass


        ################ DEFAULT PHASES ################

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                # forward
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, real_cond, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution)
                
                ######## adversarial ########

                # mask ortho visible
                if self.lossmask_mode_adv!='none':
                    sm = self.lossmask_mode_adv.split('_')
                    if sm[0]=='replace':
                        gen_img_foradv = {
                            'image': torch.lerp(real_img['image'], gen_img['image'], lmask_adv),
                            'image_raw': torch.lerp(real_img['image_raw_noblur'], gen_img['image_raw'], lmask_adv_raw),
                        }
                    else:
                        assert 0
                else:
                    gen_img_foradv = gen_img
                # uutil.pdump({k:v.detach().clone().cpu() for k,v in real_img.items()}, '/dev/shm/real_img.pkl')
                # uutil.pdump({k:v.detach().clone().cpu() for k,v in gen_img_foradv.items()}, '/dev/shm/gen_img_foradv.pkl')
                # uutil.pdump({
                #     'lmask': lmask.cpu(),
                #     'lmask_adv': lmask_adv.cpu(),
                #     'lmask_adv_raw': lmask_adv_raw.cpu(),
                #     'lmask_recon': lmask_recon.cpu(),
                #     'lmask_recon_raw': lmask_recon_raw.cpu(),
                # }, '/dev/shm/lmasks.pkl')

                gen_logits = self.run_D(gen_img_foradv, gen_c, real_cond, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                training_stats.report('Loss/G/loss', loss_Gmain)

                ######## reconstruction ########

                if self.lossmask_mode_recon!='none':
                    # xin = {
                    #     'z': gen_z,
                    #     'cond': real_cond,
                    #     'camera_params': real_cond['image_camera'],
                    #     # 'elevations': torch.zeros(len(gen_z), device=gen_z.device),
                    #     # 'azimuths': torch.zeros(len(gen_z), device=gen_z.device),
                    #     # 'distances': torch.ones(len(gen_z), device=gen_z.device),  # forgive me for I have sinned
                    # }
                    # out = self.G.f(xin)
                    out = {
                        'image': torch.lerp(real_img['image'], gen_img['image'], lmask_recon)*0.5+0.5,
                        'image_raw': torch.lerp(real_img['image_raw_noblur'], gen_img['image_raw'], lmask_recon_raw)*0.5+0.5,
                        **{k:v for k,v in gen_img.items() if k not in {'image', 'image_raw'}},
                    }
                    # uutil.pdump({k:v.detach().clone().cpu() for k,v in out.items()}, '/dev/shm/out.pkl')

                    # calc rgb losses
                    loss_recon_lpips = self.lpips_model(out['image'], real_cond['image']).mean()
                    loss_recon_l1 = (out['image'] - real_cond['image']).abs().mean()
                    training_stats.report('Loss/G/recon/lpips', loss_recon_lpips)
                    training_stats.report('Loss/G/recon/l1', loss_recon_l1)

                    # calc alpha loss
                    s = out['image_weights'].shape[-1]
                    k = 2
                    gt_alpha = real_cond['image_alpha']
                    gt_alpha = torch.nn.functional.interpolate(gt_alpha, s, mode='bilinear')
                    msk = torch.nn.functional.conv2d(
                        gt_alpha,
                        torch.ones(1,1, 2*k+1,2*k+1).to(gt_alpha.device),
                        stride=1,
                        padding=k,
                    ) / (2*k+1)**2
                    msk = (msk-0.5).abs()*2 > 0.5
                    loss_recon_alpha_l2 = (out['image_weights']-gt_alpha).pow(2).mul(msk.float() * lmask_recon_raw).mean()
                    training_stats.report('Loss/G/recon/alpha_l2', loss_recon_alpha_l2)

                    # calc depthz loss
                    gt_xyz = real_cond['image_xyz']
                    gt_xyz = torch.nn.functional.interpolate(gt_xyz, s, mode='bilinear')
                    with torch.no_grad():
                        mskz = msk & (out['image_weights'] > 0.5) & (gt_alpha>0.5)
                    loss_recon_depth_l2 = (out['image_xyz'] - gt_xyz).pow(2).sum(dim=1).sqrt().mul(mskz.float() * lmask_recon_raw).mean()
                    training_stats.report('Loss/G/recon/depth_l2', loss_recon_depth_l2)

                    # aggregate
                    loss_Grecon = (
                        self.lambda_recon_lpips * loss_recon_lpips +
                        self.lambda_recon_l1 * loss_recon_l1 +
                        self.lambda_recon_alpha_l2 * loss_recon_alpha_l2 +
                        self.lambda_recon_depth_l2 * loss_recon_depth_l2
                    )
                    training_stats.report('Loss/G/loss_recon', loss_Grecon)
                else:
                    loss_Grecon = torch.zeros_like(loss_Gmain)
                # exit(0)

            with torch.autograd.profiler.record_function('Gmain_backward'):
                (loss_Gmain.mean().mul(gain) + loss_Grecon.mean()).backward()

        # Density Regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'l1':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, real_cond, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, real_cond, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * self.G.rendering_kwargs['density_reg_p_dist']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, real_cond, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-detach':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, real_cond, update_emas=False)

            initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1 # Front

            perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, real_cond, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            monotonic_loss = torch.relu(sigma_initial.detach() - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()


            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, real_cond, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, real_cond, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, real_cond, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Alternative density regularization
        if phase in ['Greg', 'Gboth'] and self.G.rendering_kwargs.get('density_reg', 0) > 0 and self.G.rendering_kwargs['reg_type'] == 'monotonic-fixed':
            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, real_cond, update_emas=False)

            initial_coordinates = torch.rand((ws.shape[0], 2000, 3), device=ws.device) * 2 - 1 # Front

            perturbed_coordinates = initial_coordinates + torch.tensor([0, 0, -1], device=ws.device) * (1/256) * self.G.rendering_kwargs['box_warp'] # Behind
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, real_cond, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            monotonic_loss = torch.relu(sigma_initial - sigma_perturbed).mean() * 10
            monotonic_loss.mul(gain).backward()


            if swapping_prob is not None:
                c_swapped = torch.roll(gen_c.clone(), 1, 0)
                c_gen_conditioning = torch.where(torch.rand([], device=gen_c.device) < swapping_prob, c_swapped, gen_c)
            else:
                c_gen_conditioning = torch.zeros_like(gen_c)

            ws = self.G.mapping(gen_z, c_gen_conditioning, real_cond, update_emas=False)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, real_cond, update_emas=False)[:, cutoff:]
            initial_coordinates = torch.rand((ws.shape[0], 1000, 3), device=ws.device) * 2 - 1
            perturbed_coordinates = initial_coordinates + torch.randn_like(initial_coordinates) * (1/256) * self.G.rendering_kwargs['box_warp']
            all_coordinates = torch.cat([initial_coordinates, perturbed_coordinates], dim=1)
            sigma = self.G.sample_mixed(all_coordinates, torch.randn_like(all_coordinates), ws, real_cond, update_emas=False)['sigma']
            sigma_initial = sigma[:, :sigma.shape[1]//2]
            sigma_perturbed = sigma[:, sigma.shape[1]//2:]

            TVloss = torch.nn.functional.l1_loss(sigma_initial, sigma_perturbed) * self.G.rendering_kwargs['density_reg']
            TVloss.mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, real_cond, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, real_cond, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits)
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw}

                real_logits = self.run_D(real_img_tmp, real_c, real_cond, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                            r1_grads_image_raw = r1_grads[1]
                        r1_penalty = r1_grads_image.square().sum([1,2,3]) + r1_grads_image_raw.square().sum([1,2,3])
                    else: # single discrimination
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                        r1_penalty = r1_grads_image.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()


#----------------------------------------------------------------------------





