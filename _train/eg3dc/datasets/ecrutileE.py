




from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
from _util.twodee_v1 import * ; import _util.twodee_v1 as u2d

import _util.training_v1 as utraining
import _train.eg3dc.util.eg3dc_v0 as ueg3d
from _databacks import lustrous_renders_v1 as dklustr



# class DatasetWrapper(torch.utils.data.Dataset):
#     def __init__(self, subset, size=512, mirror=True, **super_kwargs):
#         self.mirror = mirror
#         self.ds = Dataset(args=Dict(prep=Dict(
#             size=size,
#             subset=subset,
#         )), split='train', collate=False)
#         return
#     def __getitem__(self, idx):
#         return self.ds[idx].to_dict()

from training.dataset import Dataset as DatasetEG3D
class DatasetWrapper(DatasetEG3D):
    def __init__(self, subset, size=512, mirror=True, **super_kwargs):
        self.mirror = mirror
        self.ds = Dataset(args=Dict(prep=Dict(
            size=size,
            subset=subset,
        )), split='train', collate=False)

        # for inheritance
        super().__init__(
            name=subset,
            raw_shape=(len(self.ds)*(int(mirror)+1), 3, size, size),
            max_size=None,
            use_labels=True,
            xflip=False,
            random_seed=0,
        )
        return

    def __getitem__(self, idx):
        idx = self._raw_idx[idx]
        x = self.ds[idx % len(self.ds)]
        img = (x['image'] * 255).type(torch.uint8).numpy()
        xyz,alpha = x['xyz'].numpy(), x['alpha'].numpy()
        imgorig = x['image'].numpy()
        assert isinstance(img, np.ndarray)
        assert list(img.shape) == self.image_shape
        assert img.dtype == np.uint8

        imgo = x['image_ortho_front'].numpy()
        imgoxyz = x['image_ortho_front_xyz'].numpy()
        imgoa = x['image_ortho_front_alpha'].numpy()
        camo = x['image_ortho_front_camera_label'].numpy()

        imgl = x['image_ortho_left'].numpy()
        imglxyz = x['image_ortho_left_xyz'].numpy()
        imgla = x['image_ortho_left_alpha'].numpy()
        caml = x['image_ortho_left_camera_label'].numpy()

        imgr = x['image_ortho_right'].numpy()
        imgrxyz = x['image_ortho_right_xyz'].numpy()
        imgra = x['image_ortho_right_alpha'].numpy()
        camr = x['image_ortho_right_camera_label'].numpy()

        imgb = x['image_ortho_back'].numpy()
        imgbxyz = x['image_ortho_back_xyz'].numpy()
        imgba = x['image_ortho_back_alpha'].numpy()
        camb = x['image_ortho_back_camera_label'].numpy()

        imgdl = x['image_dorthoA_left'].numpy()
        camdl = x['image_dorthoA_left_camera_label'].numpy()

        imgdr = x['image_dorthoA_right'].numpy()
        camdr = x['image_dorthoA_right_camera_label'].numpy()

        if idx>=len(self.ds):
            assert self.mirror
            img = img[...,::-1]
            imgorig = imgorig[...,::-1]
            xyz = xyz[...,::-1]
            xyz[0] *= -1  # flip x dim
            alpha = alpha[...,::-1]

            imgo = imgo[...,::-1]
            imgoxyz = imgoxyz[...,::-1]
            imgoxyz[0] *= -1  # flip x dim
            imgoa = imgoa[...,::-1]

            imgl = imgl[...,::-1]
            imglxyz = imglxyz[...,::-1]
            imglxyz[0] *= -1  # flip x dim
            imgla = imgla[...,::-1]

            imgr = imgr[...,::-1]
            imgrxyz = imgrxyz[...,::-1]
            imgrxyz[0] *= -1  # flip x dim
            imgra = imgra[...,::-1]

            imgb = imgb[...,::-1]
            imgbxyz = imgbxyz[...,::-1]
            imgbxyz[0] *= -1  # flip x dim
            imgba = imgba[...,::-1]

            imgdl = imgdl[...,::-1]
            imgdr = imgdr[...,::-1]

            (imgl,imglxyz,imgla),(imgr,imgrxyz,imgra) = (imgr,imgrxyz,imgra),(imgl,imglxyz,imgla)
            (imgdl,),(imgdr,) = (imgdr,),(imgdl,)
            resfeats = x['resnet_feats'][1].numpy()
            reschonk = x['resnet_chonk'][1].numpy()
        else:
            resfeats = x['resnet_feats'][0].numpy()
            reschonk = x['resnet_chonk'][0].numpy()
        label = self.get_label(idx)
        return {
            'image': img.copy(),
            'xyz': xyz.copy(),
            'alpha': alpha.copy(),
            'camera': label.copy(),
            # 'camera': label[:25],
            # 'resnet_feats': label[25:],
            'condition': {
                # 'resnet_feats': x['resnet_feats'].numpy().copy(),
                'resnet_feats': resfeats.copy(),
                'resnet_chonk': reschonk.copy(),

                'image': imgorig.copy(),
                'image_xyz': xyz.copy(),
                'image_alpha': alpha.copy(),
                'image_camera': label.copy(),
                # 'image_camera': label[:25],

                'image_ortho_front': imgo.copy(),
                'image_ortho_front_xyz': imgoxyz.copy(),
                'image_ortho_front_alpha': imgoa.copy(),
                'image_ortho_front_camera': camo.copy(),

                'image_ortho_left': imgl.copy(),
                'image_ortho_left_xyz': imglxyz.copy(),
                'image_ortho_left_alpha': imgla.copy(),
                'image_ortho_left_camera': caml.copy(),

                'image_ortho_right': imgr.copy(),
                'image_ortho_right_xyz': imgrxyz.copy(),
                'image_ortho_right_alpha': imgra.copy(),
                'image_ortho_right_camera': camr.copy(),

                'image_ortho_back': imgb.copy(),
                'image_ortho_back_xyz': imgbxyz.copy(),
                'image_ortho_back_alpha': imgba.copy(),
                'image_ortho_back_camera': camb.copy(),

                'image_dorthoA_left': imgdl.copy(),
                'image_dorthoA_left_camera': camdl.copy(),

                'image_dorthoA_right': imgdr.copy(),
                'image_dorthoA_right_camera': camdr.copy(),
            },
        }

    # def __getitem__(self, idx):
    #     image, imgo, imgoxyz, imgoa, camo = self._load_raw_image(self._raw_idx[idx])
    #     assert isinstance(image, np.ndarray)
    #     assert list(image.shape) == self.image_shape
    #     assert image.dtype == np.uint8
    #     # if self._xflip[idx]:
    #     #     assert image.ndim == 3 # CHW
    #     #     image = image[:, :, ::-1]
    #     return {
    #         'image': image.copy(),
    #         'camera': self.get_label(idx),
    #         'condition': {
    #             'image_ortho_front': imgo.copy(),
    #             'image_ortho_front_xyz': imgoxyz.copy(),
    #             'image_ortho_front_alpha': imgoa.copy(),
    #             'image_ortho_front_camera': camo.copy(),
    #         },
    #     }
    # def _load_raw_image(self, raw_idx):
    #     idx = raw_idx
    #     x = self.ds[idx % len(self.ds)]
    #     img = (x['image'] * 255).type(torch.uint8).numpy()
    #     imgo = x['image_ortho_front'].numpy()
    #     imgoxyz = x['image_ortho_front_xyz'].numpy()
    #     imgoa = x['image_ortho_front_alpha'].numpy()
    #     camo = x['image_ortho_front_camera_label'].numpy()
    #     if idx>=len(self.ds):
    #         assert self.mirror
    #         img = img[...,::-1]
    #         imgo = imgo[...,::-1]
    #         imgoxyz = imgoxyz[...,::-1]
    #         imgoxyz[0] *= -1  # flip x dim
    #         imgoa = imgoa[...,::-1]
    #     return img, imgo, imgoxyz, imgoa, camo
    def _load_raw_labels(self):
        labs = self.ds.get_all_labels()
        if self.mirror:
            labs2 = labs.copy()
            labs2[:,[1,2,3,4,8]] *= -1
            labs = np.concatenate([labs, labs2])
        return labs


    # def close(self):
    #     pass
    # def __getstate__(self):
    #     return dict(self.__dict__, ds=None)
    # def __del__(self):
    #     try:
    #         self.close()
    #     except:
    #         pass
    # def __len__(self):
    #     return len(self.ds) * (1,2)[self.xflip]
    # def __getitem__(self, idx):
    #     x = self.ds[idx % len(self.ds)]
    #     img = (x['image'] * 255).uint8().numpy()
    #     lab = x['camera_label'].numpy()
    #     if idx>=len(self.ds):
    #         assert self.xflip
    #         img = img[...,::-1]
    #         lab[1,2,3,4,8] *= -1
    #     return img, lab

    # def get_details(self, idx):
    #     return Dict({
    #         'raw_idx': idx % len(self.ds),
    #         'xflip': idx >= len(self.ds),
    #         'raw_label': self.ds[idx%len(self.ds)]['camera_label'].numpy(),
    #     })


class Dataset(torch.utils.data.Dataset):
    default_args=Dict(
        prep=Dict(
            module=utraining.infer_module_dataset(uutil.fnstrip(__file__)).dataset_name,
            size=512,
            subset='rutileEA',
            n_generations=8,
            boxwarp=0.7,
            bs=1,
        ),
    )
    def __init__(self, args=None, dk=None, split=None, collate=True, device=None):
        args = args or Dict()
        args.load.dtypes = set((
            'image', 'render_params',
        ))
        self.args_user = copy.deepcopy(args or Dict())
        self.dk = dk or dklustr.DatabackendMinna(args=args, collate=False)
        self.args = copy.deepcopy(self.default_args)
        self.args.update(self.dk.args)
        self.args.update(args or Dict())
        self.size = self.args.prep.size
        self.collate = collate
        self.device = device

        # self.split = split or 'val'
        split = split or 'val'
        if os.environ['MACHINE_NAME']=='z97x':
            self.split = split if split.startswith('val') else 'val'
        else:
            self.split = split
        self.subset = self.args.prep.subset
        self.bns = uutil.safe_bns([
            f'rutileE/rgb/{bn[-1]}/{bn}/{i:04d}'
            for bn in uutil.read_bns(
                f'{self.dk.args.base.dn}/_data/lustrous/subsets/{self.subset}_{self.split}.csv',
                safe=False,
            )
            for i in range(self.args.prep.n_generations)
        ])
        # self.resnet_feats = uutil.pload(
        #     # f'{self.dk.args.base.dn}/_data/lustrous/preprocessed/minna_resnet_feats_ortho/features_pca.pkl'
        #     f'{self.dk.args.base.dn}/_data/lustrous/renders/rutileE/minna_resnet_feats_ortho/features_pca.pkl'
        # )
        # # self.resnet_feats['features'] = torch.tensor(self.resnet_feats['features'])
        # self.resnet_feats['bn_map'] = {}
        # for i,bn in enumerate(self.resnet_feats['bns']):
        #     rs,dt,franch,idx,view = bn.split('/')
        #     self.resnet_feats['bn_map'][f'{rs}/{franch}/{idx}'] = i
        # del self.resnet_feats['bns']  # get rid of list?
        return
    def to(self, device):
        self.device = device
        return self
    def __len__(self):
        return len(self.bns)
    def get_all_labels(self):
        dk = self.dk
        ans = []
        for bn in uutil.unsafe_bns(self.bns):
            # rs,dt,franch,idx,view = bn.split('/')
            cam = dklustr.camera_params_to_matrix(
                'eg3d_lustrousB',
                **dk.rp_meta[bn]['render_params'],
            )
            ans.append(torch.cat([
                cam['matrix_extrinsic'].flatten(),
                cam['matrix_intrinsic'].flatten(),
                # self.resnet_feats['features'][self.resnet_feats['bn_map'][f'{rs}/{franch}/{idx}']],
            ]))
        return torch.stack(ans).numpy().astype(np.float32)
    def __getitem__(self, idx, collate=None, det=True, return_more=False):
        if type(idx) in (list, slice, range):
            return utorch.default_collate([
                self.__getitem__(i, collate=False, det=det, return_more=False)
                for i in uutil.idxs2list(idx, n=len(self))
            ], device=self.device)
        else:
            bw = self.args.prep.boxwarp
            bn = uutil.unsafe_bn(idx, bns=self.bns)

            # # special case fandom_align
            # rs,dtype,franch,idx,view = bn.split('/')
            # if rs=='daredemoE' and dtype=='fandom_align' and view=='front':
            #     # collect like ortho, then replace condition at end
            #     isfan = True
            #     bn = f'{rs}/ortho/{franch}/{idx}/{view}'
            # else:
            #     isfan = False

            # special case fandom_align
            rs,dtype,franch,idx,view = bn.split('/')
            isfan = rs=='daredemoE' and dtype=='fandom_align' and view=='front'
            # isfan = rs=='daredemoE' and dtype.startswith('fandom_align')
            if isfan:
                bn_original = bn
                bn = f'{rs}/ortho/{franch}/{idx}/front'

            # grab base info
            x = self.dk.__getitem__(bn, collate=False, return_more=return_more)
            # det = (self.split!='train') if det is None else det
            cam = dklustr.camera_params_to_matrix('eg3d_lustrousB', **x['render_params'])
            rs,dtype,franch,idx,view = bn.split('/')
            # dtype = 'xyza' if not rs.startswith('daredemo') else 'xyza60'
            dtype = {
                ('daredemoE', 'rgb60'): 'xyza60',
                ('daredemoE', 'ortho'): 'ortho_xyza',
            }.get((rs,dtype), 'xyza')
            xox = self.dk.__getitem__(f'{rs}/{dtype}/{franch}/{idx}/{view}', collate=False, return_more=return_more)
            xox = I(xox['image']).resize(self.size).t()
            ret = Dict({
                'bn': x['bn'],
                'image': I(x['image']).resize(self.size).convert('RGBA').bg('w').convert('RGB').t(),
                'xyz': xox[:3] * bw - bw/2,
                'alpha': xox[-1:],
                'camera_label': cam['camera_label'],
                # 'camera_label': torch.cat([
                #     cam['matrix_extrinsic'].flatten(),
                #     cam['matrix_intrinsic'].flatten(),
                # ]),
                # 'resnet_feats': torch.tensor(
                #     self.resnet_feats['features'][self.resnet_feats['bn_map'][f'{rs}/{franch}/{idx}']]
                # ),
                'resnet_feats': torch.tensor(uutil.pload(
                    f'{self.dk.args.base.dn}/_data/lustrous/renders/{rs}/ortho_katepca/{franch}/{idx}/front.pkl'
                )),
                'resnet_chonk': torch.tensor(uutil.pload(
                    f'{self.dk.args.base.dn}/_data/lustrous/renders/{rs}/ortho_katepca_chonk/{franch}/{idx}/front.pkl'
                )),
            })

            # ortho images + xyza
            rs,_,franch,idx,_ = bn.split('/')
            for view in ['front', 'left', 'right', 'back']:
                dtype = 'ortho'
                xo = self.dk.__getitem__(f'{rs}/{dtype}/{franch}/{idx}/{view}', collate=False, return_more=return_more)
                camo = dklustr.camera_params_to_matrix('eg3d_lustrousB', **xo['render_params'])
                ret[f'image_ortho_{view}'] = I(xo['image']).resize(self.size).convert('RGBA').bg('w').convert('RGB').t()
                ret[f'image_ortho_{view}_camera_label'] = camo['camera_label']
                dtype = 'ortho_xyza'
                xox = self.dk.__getitem__(f'{rs}/{dtype}/{franch}/{idx}/{view}', collate=False, return_more=return_more)
                xox = I(xox['image']).resize(self.size).t()
                ret[f'image_ortho_{view}_xyz'] = xox[:3] * bw - bw/2
                ret[f'image_ortho_{view}_alpha'] = xox[-1:]

            # dorthoA images
            rs,_,franch,idx,_ = bn.split('/')
            for view in ['left', 'right', ]:
                dtype = 'dorthoA'
                xo = self.dk.__getitem__(f'{rs}/{dtype}/{franch}/{idx}/{view}', collate=False, return_more=return_more)
                camo = dklustr.camera_params_to_matrix('eg3d_lustrousB', **xo['render_params'])
                ret[f'image_dorthoA_{view}'] = I(xo['image']).resize(self.size).t()
                ret[f'image_dorthoA_{view}_camera_label'] = camo['camera_label']

            # special case fandom_align
            if isfan:
                ret['bn'] = bn_original
                rs,dtype,franch,idx,view = bn_original.split('/')
                xo = self.dk.__getitem__(bn_original, collate=False, return_more=return_more)
                ret['resnet_feats'] = torch.tensor(uutil.pload(
                    f'{self.dk.args.base.dn}/_data/lustrous/renders/{rs}/fandom_align_katepca/{franch}/{idx}/front.pkl'
                ))
                ret['resnet_chonk'] = torch.tensor(uutil.pload(
                    f'{self.dk.args.base.dn}/_data/lustrous/renders/{rs}/fandom_align_katepca_chonk/{franch}/{idx}/front.pkl'
                ))
                ret[f'image_ortho_front'] = I(xo['image']).resize(self.size).convert('RGBA').bg('w').convert('RGB').t()

            # # augmentations
            # _flip_h = (not det) and (np.random.rand()<0.5)
            # if _flip_h:
            #     ret['image'] = ret['image'].flip(dims=(-1,))
            #     ret['camera_params'] = dkmodlust.mirror_label(ret['camera_params'])

            # if :
            #     ret['']

            # boilerplate
            if (collate is None and self.collate) or collate:
                ret = utorch.default_collate([ret,])
            ret = utorch.to(ret, self.device)
            if return_more: ret.update({'locals': locals()})
            return ret

class Datamodule(pl.LightningDataModule):
    default_args=Dict(
        **dklustr.DatabackendMinna.default_args,
        **Dataset.default_args,
        train=Dict(
            machine=os.environ.get('MACHINE_NAME'),
            gpus=torch.cuda.device_count(),
            max_bs_per_gpu=None,
            num_workers='max',
        ),
    )
    def __init__(self, args=None):
        super().__init__()
        args = args or Dict()
        args.load.dtypes = set((
            'image', 'render_params',
        ))
        self.args_user = copy.deepcopy(args or Dict())
        self.dk = dklustr.DatabackendMinna(args, collate=False)
        self.args = copy.deepcopy(self.default_args)
        self.args.update(self.dk.args)
        self.args.update(args or Dict())

        ibs = utraining.infer_batch_size(
            self.args.prep.bs,
            self.args.train.gpus,
            self.args.train.max_bs_per_gpu,
        )
        self.bs_realized = ibs['bs_realized']
        self.accumulate_grad_batches = ibs['accumulate_grad_batches']
        self.num_workers = utraining.infer_num_workers(self.args.train.num_workers)
        return
    def dataloader(self, split='val', shuffle=False):
        ds = Dataset(
            args=self.args,
            dk=self.dk,
            split=split,
            collate=False,
            device=None,
        )
        dl = torch.utils.data.DataLoader(
            ds, batch_size=self.bs_realized,
            shuffle=shuffle, num_workers=self.num_workers,
            drop_last=False,
        )
        return dl
    def train_dataloader(self):
        return self.dataloader('train', shuffle=True)
    def val_dataloader(self):
        return self.dataloader('val', shuffle=False)
    def test_dataloader(self):
        return self.dataloader('test', shuffle=False)

















