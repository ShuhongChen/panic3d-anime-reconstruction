


from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
from _util.twodee_v1 import * ; import _util.twodee_v1 as u2d
from _util.threedee_v0 import * ; import _util.threedee_v0 as u3d
from _util.video_v1 import * ; import _util.video_v1 as uvid

# vanilla
vanilla = '6152365338188306398'

# render angles
cam60 = torch.tensor(np.stack(np.meshgrid(
    np.linspace(60, -20,  5),
    np.linspace(-180, 150, 12),
)).T.reshape(60,-1)).float()
camsubs = {
    'all': list(range(60)),
    'front1': [42,],
    'front15': [
        28, 29, 30, 31, 32,
        40, 41, 42, 43, 44,
        52, 53, 54, 55, 56,
    ],
    'spin12': [
        *range(42, 48),
        *range(36, 42),
    ],
}

# camera converter
def camera_params_to_matrix(mode, **kwargs):
    if mode=='eg3d_lustrousB':
        fov = kwargs['fov']
        elev = kwargs['elev']
        azim = kwargs['azim']# + 180
        dist = kwargs['dist']
        if isinstance(fov, torch.Tensor): fov = fov.item()
        if isinstance(elev, torch.Tensor): elev = elev.item()
        if isinstance(azim, torch.Tensor): azim = azim.item()
        if isinstance(dist, torch.Tensor): dist = dist.item()
        
        focal_length = 0.5/np.tan((fov/2)*np.pi/180)
        mat_intr = np.asarray([
            [focal_length, 0, 0.5,],
            [0, focal_length, 0.5,],
            [0, 0, 1,],
        ], dtype=np.float32)
        R = np.eye(4)
        R[:3,:3] = scipy.spatial.transform.Rotation.from_euler(
            'xyz', [elev,azim,0], degrees=True,
        ).as_matrix().T
        R[[0,2]] *= -1
        R[2,-1] = -dist
        mat_extr = np.asarray([
            [-1,0,0,0],
            [0,1,0,0],
            [0,0,-1,0],
            [0,0,0,1],
        ]) @ np.linalg.inv(R) @ np.asarray([
            [1,0,0,0],
            [0,-1,0,0],
            [0,0,-1,0],
            [0,0,0,1],
        ])
        mat_intr = torch.tensor(mat_intr).float()
        mat_extr = torch.tensor(mat_extr).float()
        return Dict(
            matrix_intrinsic=mat_intr,
            matrix_extrinsic=mat_extr,
            camera_label=torch.cat([mat_extr.flatten(), mat_intr.flatten()]),
        )
    else:
        assert 0, 'mode not understood'

# ortho ray helper
def get_rays_ortho(elev, azim, dist, boxwarp, resolution, device=None):
    e,a,d,bw,r = elev, azim, dist, boxwarp, resolution
    if isinstance(e, torch.Tensor): e = e.item()
    if isinstance(a, torch.Tensor): a = a.item()
    mg = torch.arange(r, device=device)
    mg = torch.stack(torch.meshgrid(
        ( (mg + 0.5) / r * bw - bw/2 ),
        -( (mg + 0.5) / r * bw - bw/2 ),
        indexing='xy',
    )+(
        torch.zeros(r,r, device=device),
        # torch.ones(r,r, device=device),
    ))
    mg = torch.stack([
        mg,
        mg + torch.tensor([0.0,0.0,-1.0,], device=device)[:,None,None],
    ])
    mg[:,2] += d  # translate by distance
    rot = torch.tensor(scipy.spatial.transform.Rotation.from_euler(
        'xyz', [-e, a, 0.0], degrees=True,
    ).as_matrix(), device=device, dtype=mg.dtype)
    t = (rot @ mg.permute(0,2,3,1)[...,None]).permute(-1,0,3,1,2)[0]
    force_rays = Dict({
        'ray_origins': t[0][None],
        'ray_directions': (t[1]-t[0])[None],
    })
    return force_rays


class DatabackendMinna:
    default_args=Dict(
        base=Dict(dn='.', project=os.environ.get('PROJECT_NAME')),
        load=Dict(dtypes=None),  # dtypes=None ==> all
    )
    def __init__(self, args=None, collate=False):
        self.args_user = copy.deepcopy(args or Dict())
        self.args = copy.deepcopy(self.default_args); self.args.update(args or Dict())
        self.collate = collate
        self.dn = f'{self.args.base.dn}/_data/lustrous'
        self.bns = uutil.safe_bns(self.get_bns())
        self.dtypes = set(self.args.load.dtypes) if self.args.load.dtypes!=None else set((
            'image', 'render_params',
        ))
        if 'render_params' in self.dtypes:
            self.rp_meta = {}
            for mfn in [
                f'{self.dn}/renders/rutileE/rutileE_meta.json',
                f'{self.dn}/renders/daredemoE/daredemoE_meta.json',
                f'{self.dn}/renders/daredemoE/danbooru_rutileE/renderparams.json',
                f'{self.dn}/renders/kiddoWE/danbooru_rutileE/renderparams.json',
                f'{self.dn}/renders/daredemoE/asoul_meta.json',
            ]:
                if os.path.isfile(mfn):
                    self.rp_meta.update(uutil.jread(mfn))

            # fuckin whatever man
            for k,v in list(self.rp_meta.items()):
                if k.startswith('rutileE/ortho/'):
                    self.rp_meta[k.replace('rutileE/ortho', 'rutileE/ortho_xyza')] = v
                    self.rp_meta[k.replace('rutileE/ortho', 'rutileE/dorthoA')] = v
                elif k.startswith('daredemoE/ortho/'):
                    self.rp_meta[k.replace('daredemoE/ortho', 'daredemoE/ortho_xyza')] = v
                    self.rp_meta[k.replace('daredemoE/ortho', 'daredemoE/fandom_align')] = v
                    self.rp_meta[k.replace('daredemoE/ortho', 'daredemoE/fandom_align_rmlineEA')] = v
                    self.rp_meta[k.replace('daredemoE/ortho', 'daredemoE/dorthoA')] = v
            ov = {'render_params': {
                'elev': 0.0,
                'azim': 0.0,
                'dist': 1.0,
                'fov': -1,
                'near': 0.5,
                'far': 1.5,
                'boxwarp': 0.7,
            }}
            for bn in uutil.unsafe_bns(self.bns):
                if bn.startswith('virtualyoutuberE/'):
                    self.rp_meta[bn] = ov
            # for p,d,f in os.walk(f'{self.dn}/renders/virtualyoutuberE'):
            #     # if p.split('/')[-1][0]=='_': continue
            #     for fn in f:
            #         if fn.endswith('.png'):
            #             sp = f'{p}/{fn}'.split('/')[-5:]
            #             if len(sp)!=5: continue
            #             rs,dtype,franch,idx,view_ = sp
            #             view = fnstrip(view_)
            #             if dtype[0]=='_': continue
            #             self.rp_meta[f'{rs}/{dtype}/{franch}/{idx}/{view}'] = ov
        return
    def __len__(self):
        return len(self.bns)
    def __getitem__(self, bn, collate=None, return_more=False):
        bn = uutil.unsafe_bn(bn, bns=self.bns)
        rs,dtype,franch,idx,view = bn.split('/')
        ret = Dict({
            'bn': bn,
            'info': {
                'renderset': rs,
                'dtype': dtype,
                'franch': franch,
                'idx': idx,
                'view': view,
            },
        })

        if 'image' in self.dtypes:
            ret['image'] = I(f'{self.dn}/renders/{bn}.png')
        if 'render_params' in self.dtypes:
            ret['render_params'] = self.rp_meta[bn]['render_params']
        
        # boilerplate
        if (collate is None and self.collate) or collate:
            ret = Dict(torch.utils.data.dataloader.default_collate([ret,]))
        if return_more: ret.update({'locals': locals()})
        return ret
    def get_bns(self):
        return sorted([
            f'{renderset}/{dtype}/{franch}/{idx}/{fnstrip(viewfn)}'
            for renderset in ['rutileE', 'daredemoE', 'virtualyoutuberE']
            if os.path.isdir(f'{self.dn}/renders/{renderset}')
            for dtype in os.listdir(f'{self.dn}/renders/{renderset}')
            # for dtype in ['rgb', 'xyza', 'ortho', 'rgb60', 'xyza60']
            if os.path.isdir(f'{self.dn}/renders/{renderset}/{dtype}')
            for franch in os.listdir(f'{self.dn}/renders/{renderset}/{dtype}')
            if franch[0]!='_' and os.path.isdir(f'{self.dn}/renders/{renderset}/{dtype}/{franch}')
            for idx in os.listdir(f'{self.dn}/renders/{renderset}/{dtype}/{franch}')
            if os.path.isdir(f'{self.dn}/renders/{renderset}/{dtype}/{franch}/{idx}')
            for viewfn in os.listdir(f'{self.dn}/renders/{renderset}/{dtype}/{franch}/{idx}')
            if viewfn.endswith('.png') and viewfn[0]!='_'
        ])













