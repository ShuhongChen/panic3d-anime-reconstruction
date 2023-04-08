


from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
from _util.twodee_v1 import * ; import _util.twodee_v1 as u2d


class DatabackendLustrousPhosA:
    default_args=Dict(
        base=Dict(dn='.', project=os.environ.get('PROJECT_NAME')),
        load=Dict(dtypes=None),  # dtypes=None ==> all
    )
    def __init__(self, args=None, collate=False):
        self._phos_type = 'phosA'
        self.args_user = copy.deepcopy(args or Dict())
        self.args = copy.deepcopy(self.default_args); self.args.update(args or Dict())
        self.collate = collate
        self.dn = f'{self.args.base.dn}/_data/lustrous'
        self.bns = uutil.safe_bns(self.get_bns())
        self.dtypes = set(self.args.load.dtypes) if self.args.load.dtypes!=None else set((
            'image',
        ))
        return
    def __len__(self):
        return len(self.bns)
    def __getitem__(self, bn, collate=None, return_more=False):
        bn = uutil.unsafe_bn(bn, bns=self.bns)
        mid,iid = bn.split('/')
        ret = Dict({
            'bn': bn,
        })

        if 'image' in self.dtypes:
            ret['image'] = I(f'{self.dn}/renders/{self._phos_type}/512/{mid[-1]}/{mid}/{iid}.png')
        
        # boilerplate
        if (collate is None and self.collate) or collate:
            ret = Dict(torch.utils.data.dataloader.default_collate([ret,]))
        if return_more: ret.update({'locals': locals()})
        return ret
    def get_bns(self):
        return sorted([
            f'{dn}/{uutil.fnstrip(fn)}'
            for i in range(10)
            if os.path.isdir(f'{self.dn}/renders/{self._phos_type}/512/{i}')
            for dn in os.listdir(f'{self.dn}/renders/{self._phos_type}/512/{i}')
            if os.path.isdir(f'{self.dn}/renders/{self._phos_type}/512/{i}/{dn}')
            for fn in os.listdir(f'{self.dn}/renders/{self._phos_type}/512/{i}/{dn}')
            if fn.endswith('.png')
        ])

class DatabackendLustrousPhosB:
    default_args=Dict(
        base=Dict(dn='.', project=os.environ.get('PROJECT_NAME')),
        load=Dict(dtypes=None),  # dtypes=None ==> all
    )
    def __init__(self, args=None, collate=False):
        self._phos_type = 'phosB'
        self.args_user = copy.deepcopy(args or Dict())
        self.args = copy.deepcopy(self.default_args); self.args.update(args or Dict())
        self.collate = collate
        self.dn = f'{self.args.base.dn}/_data/lustrous'
        self.bns = uutil.safe_bns(self.get_bns())
        self.dtypes = set(self.args.load.dtypes) if self.args.load.dtypes!=None else set((
            'image',
        ))
        return
    def __len__(self):
        return len(self.bns)
    def __getitem__(self, bn, collate=None, return_more=False):
        bn = uutil.unsafe_bn(bn, bns=self.bns)
        mid,iid = bn.split('/')
        ret = Dict({
            'bn': bn,
        })

        if 'image' in self.dtypes:
            ret['image'] = I(f'{self.dn}/renders/{self._phos_type}/512/{mid[-1]}/{mid}/{iid}.png')
        
        # boilerplate
        if (collate is None and self.collate) or collate:
            ret = Dict(torch.utils.data.dataloader.default_collate([ret,]))
        if return_more: ret.update({'locals': locals()})
        return ret
    def get_bns(self):
        return sorted([
            f'{dn}/{uutil.fnstrip(fn)}'
            for i in range(10)
            if os.path.isdir(f'{self.dn}/renders/{self._phos_type}/512/{i}')
            for dn in os.listdir(f'{self.dn}/renders/{self._phos_type}/512/{i}')
            if os.path.isdir(f'{self.dn}/renders/{self._phos_type}/512/{i}/{dn}')
            for fn in os.listdir(f'{self.dn}/renders/{self._phos_type}/512/{i}/{dn}')
            if fn.endswith('.png')
        ])

class DatabackendLustrousPhosC:
    default_args=Dict(
        base=Dict(dn='.', project=os.environ.get('PROJECT_NAME')),
        load=Dict(dtypes=None),  # dtypes=None ==> all
    )
    def __init__(self, args=None, collate=False):
        self._phos_type = 'phosC'
        self.args_user = copy.deepcopy(args or Dict())
        self.args = copy.deepcopy(self.default_args); self.args.update(args or Dict())
        self.collate = collate
        self.dn = f'{self.args.base.dn}/_data/lustrous'
        self.bns = uutil.safe_bns(self.get_bns())
        self.dtypes = set(self.args.load.dtypes) if self.args.load.dtypes!=None else set((
            'image',
        ))
        return
    def __len__(self):
        return len(self.bns)
    def __getitem__(self, bn, collate=None, return_more=False):
        bn = uutil.unsafe_bn(bn, bns=self.bns)
        mid,iid = bn.split('/')
        ret = Dict({
            'bn': bn,
        })

        if 'image' in self.dtypes:
            ret['image'] = I(f'{self.dn}/renders/{self._phos_type}/512/{mid[-1]}/{mid}/{iid}.png')
        
        # boilerplate
        if (collate is None and self.collate) or collate:
            ret = Dict(torch.utils.data.dataloader.default_collate([ret,]))
        if return_more: ret.update({'locals': locals()})
        return ret
    def get_bns(self):
        return sorted([
            f'{dn}/{uutil.fnstrip(fn)}'
            for i in range(10)
            if os.path.isdir(f'{self.dn}/renders/{self._phos_type}/512/{i}')
            for dn in os.listdir(f'{self.dn}/renders/{self._phos_type}/512/{i}')
            if os.path.isdir(f'{self.dn}/renders/{self._phos_type}/512/{i}/{dn}')
            for fn in os.listdir(f'{self.dn}/renders/{self._phos_type}/512/{i}/{dn}')
            if fn.endswith('.png')
        ])

class DatabackendLustrousKongo:
    default_args=Dict(
        base=Dict(dn='.', project=os.environ.get('PROJECT_NAME')),
        load=Dict(dtypes=None),  # dtypes=None ==> all
    )
    def __init__(self, args=None, collate=False):
        # hard-code kongoC is ok b/c symlink anyway
        self._phos_type = 'kongoC'
        self.args_user = copy.deepcopy(args or Dict())
        self.args = copy.deepcopy(self.default_args); self.args.update(args or Dict())
        self.collate = collate
        self.dn = f'{self.args.base.dn}/_data/lustrous'
        self.bns = uutil.safe_bns(self.get_bns())
        self.dtypes = set(self.args.load.dtypes) if self.args.load.dtypes!=None else set((
            'image', 'camera_label', # 'image_xyz',
        ))
        if 'camera_label' in self.dtypes:
            labels = uutil.jread(
                f'{self.dn}/renders/{self._phos_type}/eg3d_labels.json'
            )
            self.camera_labels = dict(labels['labels'])
        return
    def __len__(self):
        return len(self.bns)
    def __getitem__(self, bn, collate=None, return_more=False):
        bn = uutil.unsafe_bn(bn, bns=self.bns)
        mid,iid = bn.split('/')
        ret = Dict({
            'bn': bn,
        })

        if 'image' in self.dtypes:
            ret['image'] = I(f'{self.dn}/renders/{self._phos_type}/512/{mid[-1]}/{mid}/{iid}.png')
        if 'image_xyz' in self.dtypes:
            ret['image_xyz'] = I(f'{self.dn}/renders/{self._phos_type}/xyz/{mid[-1]}/{mid}/{iid}.png')
        if 'camera_label' in self.dtypes:
            ret['camera_label'] = torch.tensor(self.camera_labels[f'{mid[-1]}/{mid}/{iid}.png'])
        
        # boilerplate
        if (collate is None and self.collate) or collate:
            ret = Dict(torch.utils.data.dataloader.default_collate([ret,]))
        if return_more: ret.update({'locals': locals()})
        return ret
    def get_bns(self):
        return sorted([
            f'{dn}/{uutil.fnstrip(fn)}'
            for i in range(10)
            if os.path.isdir(f'{self.dn}/renders/{self._phos_type}/512/{i}')
            for dn in os.listdir(f'{self.dn}/renders/{self._phos_type}/512/{i}')
            if os.path.isdir(f'{self.dn}/renders/{self._phos_type}/512/{i}/{dn}')
            for fn in os.listdir(f'{self.dn}/renders/{self._phos_type}/512/{i}/{dn}')
            if fn.endswith('.png')
        ])

class DatabackendLustrousCinnabar:
    default_args=Dict(
        base=Dict(dn='.', project=os.environ.get('PROJECT_NAME')),
        load=Dict(dtypes=None),  # dtypes=None ==> all
    )
    def __init__(self, args=None, collate=False):
        # hard-code kongoC is ok b/c symlink anyway
        self._phos_type = 'cinnabarC'
        self.args_user = copy.deepcopy(args or Dict())
        self.args = copy.deepcopy(self.default_args); self.args.update(args or Dict())
        self.collate = collate
        self.dn = f'{self.args.base.dn}/_data/lustrous'
        self.bns = uutil.safe_bns(self.get_bns())
        self.dtypes = set(self.args.load.dtypes) if self.args.load.dtypes!=None else set((
            'image', 'camera_label',
        ))
        if 'camera_label' in self.dtypes:
            labels = uutil.jread(
                f'{self.dn}/renders/{self._phos_type}/eg3d_labels.json'
            )
            self.camera_labels = dict(labels['labels'])
        return
    def __len__(self):
        return len(self.bns)
    def __getitem__(self, bn, collate=None, return_more=False):
        bn = uutil.unsafe_bn(bn, bns=self.bns)
        mid,iid = bn.split('/')
        ret = Dict({
            'bn': bn,
        })

        if 'image' in self.dtypes:
            ret['image'] = I(f'{self.dn}/renders/{self._phos_type}/512/{mid[-1]}/{mid}/{iid}.png')
        if 'camera_label' in self.dtypes:
            ret['camera_label'] = torch.tensor(self.camera_labels[f'{mid[-1]}/{mid}/{iid}.png'])
        
        # boilerplate
        if (collate is None and self.collate) or collate:
            ret = Dict(torch.utils.data.dataloader.default_collate([ret,]))
        if return_more: ret.update({'locals': locals()})
        return ret
    def get_bns(self):
        return sorted([
            f'{dn}/{uutil.fnstrip(fn)}'
            for i in range(10)
            if os.path.isdir(f'{self.dn}/renders/{self._phos_type}/512/{i}')
            for dn in os.listdir(f'{self.dn}/renders/{self._phos_type}/512/{i}')
            if os.path.isdir(f'{self.dn}/renders/{self._phos_type}/512/{i}/{dn}')
            for fn in os.listdir(f'{self.dn}/renders/{self._phos_type}/512/{i}/{dn}')
            if fn.endswith('.png')
        ])

class DatabackendLustrousDaredemo:
    default_args=Dict(
        base=Dict(dn='.', project=os.environ.get('PROJECT_NAME')),
        load=Dict(dtypes=None),  # dtypes=None ==> all
    )
    def __init__(self, args=None, collate=False):
        # hard-code kongoC is ok b/c symlink anyway
        self._phos_type = 'daredemoC'
        self.args_user = copy.deepcopy(args or Dict())
        self.args = copy.deepcopy(self.default_args); self.args.update(args or Dict())
        self.collate = collate
        self.dn = f'{self.args.base.dn}/_data/lustrous'
        self.bns = uutil.safe_bns(self.get_bns())
        self.dtypes = set(self.args.load.dtypes) if self.args.load.dtypes!=None else set((
            'image', 'camera_label', 'image_xyz',
        ))
        if 'camera_label' in self.dtypes:
            labels = uutil.jread(
                f'{self.dn}/renders/{self._phos_type}/eg3d_labels.json'
            )
            self.camera_labels = dict(labels['labels'])
        return
    def __len__(self):
        return len(self.bns)
    def __getitem__(self, bn, collate=None, return_more=False):
        bn = uutil.unsafe_bn(bn, bns=self.bns)
        franch,mid,iid = bn.split('/')
        ret = Dict({
            'bn': bn,
        })

        if 'image' in self.dtypes:
            ret['image'] = I(f'{self.dn}/renders/{self._phos_type}/512/{franch}/{mid}/{iid}.png')
        if 'image_xyz' in self.dtypes:
            ret['image_xyz'] = I(f'{self.dn}/renders/{self._phos_type}/xyz/{franch}/{mid}/{iid}.png')
        if 'camera_label' in self.dtypes:
            ret['camera_label'] = torch.tensor(self.camera_labels[f'{franch}/{mid}/{iid}.png'])
        
        # boilerplate
        if (collate is None and self.collate) or collate:
            ret = Dict(torch.utils.data.dataloader.default_collate([ret,]))
        if return_more: ret.update({'locals': locals()})
        return ret
    def get_bns(self):
        return sorted([
            f'{"/".join(p.split("/")[-2:])}/{fnstrip(fn)}'
            for p,d,f in os.walk(f'{self.dn}/renders/{self._phos_type}/512')
            for fn in f
            if fn.endswith('.png')
            # f'{dn}/{uutil.fnstrip(fn)}'
            # for i in range(10)
            # if os.path.isdir(f'{self.dn}/renders/{self._phos_type}/512/{i}')
            # for dn in os.listdir(f'{self.dn}/renders/{self._phos_type}/512/{i}')
            # if os.path.isdir(f'{self.dn}/renders/{self._phos_type}/512/{i}/{dn}')
            # for fn in os.listdir(f'{self.dn}/renders/{self._phos_type}/512/{i}/{dn}')
            # if fn.endswith('.png')
        ])










_name2dk = {
    'phosA': DatabackendLustrousPhosA,
    'phosB': DatabackendLustrousPhosB,
    'phosC': DatabackendLustrousPhosC,
    'kongoA': DatabackendLustrousKongo,
    'kongoB': DatabackendLustrousKongo,
    'kongoC': DatabackendLustrousKongo,
}

def mirror_label(lab):
    lab = copy.deepcopy(lab)
    lab[:,[1,2,3,4,8]] *= -1
    return lab

