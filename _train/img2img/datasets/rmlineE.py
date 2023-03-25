




from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
from _util.twodee_v1 import * ; import _util.twodee_v1 as u2d

import _util.training_v1 as utraining
# import _train.eg3d.util.eg3d_v0 as ueg3d
# from _databacks import lustrous_renders_v1 as dklustr


class Dataset(torch.utils.data.Dataset):
    default_args=Dict(
        prep=Dict(
            module=utraining.infer_module_dataset(uutil.fnstrip(__file__)).dataset_name,
            size=21,
            bs=1,
            augment_dilate_mask=(1,3),
        ),
    )
    def __init__(self, args=None, split=None, collate=True, device=None):
        args = args or Dict()
        args.load.dtypes = set((
            'image', 'line_mask', 'face_hull',
        ))
        self.args_user = copy.deepcopy(args or Dict())
        self.args = copy.deepcopy(self.default_args)
        # self.args.update(self.dk.args)
        self.args.update(args or Dict())
        self.size = self.args.prep.size
        assert self.size==21
        self.augment_dilate_mask = self.args.prep.augment_dilate_mask
        self.collate = collate
        self.device = device

        self.split = split or 'val'
        # split = split or 'val'
        # if os.environ['MACHINE_NAME']=='z97x':
        #     self.split = split if split.startswith('val') else 'val'
        # else:
        #     self.split = split

        # load all data
        dn = f'{self.args.base.dn or "."}/_data/lustrous/preprocessed/patches'
        if self.split=='train':
            # load train data
            self.data_render = uutil.pload(f'{dn}/rmlineERA_train.pkl')
            self.data_illust = uutil.pload(f'{dn}/rmlineERA_train.pkl')
            self.data_render['bns'] = [
                (uutil.safe_bn(fn), loc)
                for fn,loc in self.data_render['bns']
            ]
            self.data_illust['bns'] = [
                (uutil.safe_bn(fn), loc)
                for fn,loc in self.data_illust['bns']
            ]
            self.bns = uutil.safe_bns([f'{i}' for i in range(len(self.data_illust['bns']))])
        else:
            # load val data
            self.data_illust = uutil.pload(f'{dn}/rmlineEIA_test.pkl')
            self.data_illust['bns'] = [
                (uutil.safe_bn(fn), loc)
                for fn,loc in self.data_illust['bns']
            ]
            self.bns = uutil.safe_bns([f'{i}' for i in range(len(self.data_illust['bns']))])
        return
    def to(self, device):
        self.device = device
        return self
    def __len__(self):
        return len(self.bns)
    def __getitem__(self, idx, collate=None, det=None, return_more=False):
        if type(idx) in (list, slice, range):
            return utorch.default_collate([
                self.__getitem__(i, collate=False, det=det, return_more=False)
                for i in uutil.idxs2list(idx, n=len(self))
            ], device=self.device)
        else:
            if self.split=='train':
                # training set
                bn = uutil.unsafe_bn(idx, bns=self.bns)
                det = self.split!='train' if det==None else det
                rlen = len(self.data_render['bns'])
                di = int(bn)
                dr = int(bn)%rlen if det else np.random.choice(rlen)
                ret = Dict({
                    'bn': bn,
                    'image': torch.tensor(np.stack([
                        self.data_illust['images'][di],
                        self.data_render['images'][dr],
                    ])),
                    'line_mask': torch.tensor(np.stack([
                        self.data_illust['line_masks'][di],
                        self.data_render['line_masks'][dr],
                    ])),
                    'face_hull': torch.tensor(np.stack([
                        self.data_illust['face_hulls'][di],
                        self.data_render['face_hulls'][dr],
                    ])),
                    'real_label': torch.tensor([0,1]),
                })
            else:
                # testing set
                bn = uutil.unsafe_bn(idx, bns=self.bns)
                det = self.split!='train' if det==None else det
                di = int(bn)
                ret = Dict({
                    'bn': bn,
                    'image': torch.tensor(np.stack([
                        self.data_illust['images'][di],
                    ])),
                    'line_mask': torch.tensor(np.stack([
                        self.data_illust['line_masks'][di],
                    ])),
                    'face_hull': torch.tensor(np.stack([
                        self.data_illust['face_hulls'][di],
                    ])),
                    'real_label': torch.tensor([0,]),
                })

            # augment masks
            if not det:
                dil = np.random.choice(self.augment_dilate_mask)
                if dil>1:
                    ret['line_mask'] = kornia.morphology.dilation(
                        ret['line_mask'].float(),
                        torch.ones(dil,dil),
                    )


            # boilerplate
            if (collate is None and self.collate) or collate:
                ret = utorch.default_collate([ret,])
            ret = utorch.to(ret, self.device)
            if return_more: ret.update({'locals': locals()})
            return ret

class Datamodule(pl.LightningDataModule):
    default_args=Dict(
        # **dklustr.DatabackendMinna.default_args,
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
            'image', 'line_mask', 'face_hull',
        ))
        self.args_user = copy.deepcopy(args or Dict())
        # self.dk = dklustr.DatabackendMinna(args, collate=False)
        self.args = copy.deepcopy(self.default_args)
        # self.args.update(self.dk.args)
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
            # dk=self.dk,
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

















