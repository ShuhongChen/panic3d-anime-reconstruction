



from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch

try:
    import skimage
    from skimage import measure as _
    from skimage import color as _
    from skimage import segmentation as _
    from skimage import filters as _
    from scipy.spatial.transform import Rotation as _
except:
    pass
try:
    import colorsys
except:
    pass
try:
    import imagesize
except:
    pass



_FN_ARIAL = f'{os.environ["PROJECT_DN"]}/_env/arial_monospaced_mt.ttf'
if not os.path.isfile(_FN_ARIAL):
    _FN_ARIAL = f'{os.environ["PROJECT_DN"]}/env/arial_monospaced_mt.ttf'
if not os.path.isfile(_FN_ARIAL):
    _FN_ARIAL = f'{os.environ["PROJECT_DN"]}/__env__/arial_monospaced_mt.ttf'
if not os.path.isfile(_FN_ARIAL):
    _FN_ARIAL = None

class I:
    # canonize
    def __init__(self, data):
        self.data = data
        if isinstance(self.data, Image.Image):
            # pil image
            self.dtype = 'pil'
            self.mode = self.data.mode
            assert self.mode in ['L','RGB','RGBA']
            self.shape = (
                len(self.data.getbands()),
                self.data.size[1],
                self.data.size[0],
            )
            self.size = self.shape[1:]
            self.diam = np.sqrt(self.size[0]**2 + self.size[1]**2)
        elif isinstance(self.data, np.ndarray):
            # np array: float(ch,h,w)
            self.dtype = 'numpy'
            if len(self.data.shape)==2:
                self.data = self.data[None,]
            elif len(self.data.shape)==4:
                assert self.data.shape[0]==1
                self.data = self.data[0]
            if self.data.shape[0] not in [1,3,4]:
                assert self.data.shape[-1] in [1,3,4]
                self.data = self.data.transpose(2,0,1)
            if self.data.dtype==bool:
                self.data = self.data.astype(float)
            elif np.issubdtype(self.data.dtype, np.integer):
                self.data = self.data.astype(float) / 255.0
            self.mode = (None,'L',None,'RGB','RGBA')[self.data.shape[0]]
            self.shape = self.data.shape
            self.size = self.shape[1:]
            self.diam = np.sqrt(self.size[0]**2 + self.size[1]**2)
        elif isinstance(self.data, torch.Tensor):
            # torch tensor: float(ch,h,w)
            self.dtype = 'torch'
            self.data = self.data.detach().cpu()
            if len(self.data.shape)==2:
                self.data = self.data[None,]
            elif len(self.data.shape)==4:
                assert self.data.shape[0]==1
                self.data = self.data[0]
            if self.data.shape[0] not in [1,3,4]:
                assert self.data.shape[-1] in [1,3,4]
                self.data = self.data.permute(2,0,1)
            if self.data.dtype==torch.bool:
                self.data = self.data.float()
            self.mode = (None,'L',None,'RGB','RGBA')[self.data.shape[0]]
            self.shape = tuple(self.data.shape)
            self.size = self.shape[1:]
            self.diam = np.sqrt(self.size[0]**2 + self.size[1]**2)
        elif isinstance(self.data, I):
            self.dtype = self.data.dtype
            self.mode = self.data.mode
            self.shape = self.data.shape
            self.size = self.data.size
            self.diam = self.data.diam
            self.data = self.data.data
        elif isinstance(self.data, str):
            if self.data.lower().endswith('.exr'):
                raw = cv2.imread(self.data, cv2.IMREAD_UNCHANGED)
                if len(raw.shape)==3:
                    if raw.shape[-1]==3:
                        raw = raw[...,::-1]  # bgr
                    elif raw.shape[-1]==4:
                        raw = raw[...,[2,1,0,3]]  # bgra
                elif len(raw.shape)==2:
                    raw = raw  # do nothing, L
                else:
                    assert 0, 'data not understood'
                self.__init__(raw)
            elif self.data.startswith('data:image/png;base64,'):
                with urllib.request.urlopen(self.data) as response:
                    img = Image.open(io.BytesIO(response.read()))
                self.__init__(img)
            else:
                self.__init__(Image.open(self.data))
        elif isinstance(self.data, plt.Figure):
            buff = io.BytesIO()
            self.data.savefig(buff)
            self.__init__(Image.open(buff))
        else:
            assert 0, 'image data not understood'
        return
    
    # retrieval
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return I(self.tensor()[idx])
        elif isinstance(idx, str):
            idx = idx.lower()
            if len(idx)==1:
                return self['rgba'.index(idx)]
            elif idx=='rgb':
                if self.mode=='RGBA':
                    return I(self.tensor()[:3])
                else:
                    return self.convert('RGB')
            elif idx=='rgba':
                return self.convert('RGBA')
            elif idx=='l':
                return self.convert('L')
            else:
                assert 0
        else:
            assert 0, 'index not understood'

    # conversion
    def convert(self, mode):
        mode = mode.upper()
        if self.mode==mode:
            return I(self)
        else:
            return I(self.pil().convert(mode))
    def invert(self, invert_alpha=False):
        data = self.np()
        if self.mode=='RGBA' and not invert_alpha:
            return I(np.concatenate([
                1-data[:3], data[3:],
            ]))
        else:
            return I(1-self.np())
    def norm(self, low=0, high=1):
        c = self.tensor()
        c = c - c.min()
        c = c * (high-low) / c.max() + low
        return I(c)
    def fc(self, clip=None):
        assert self.mode=='L'
        d = self.torch()
        c = clip if clip!=None else d.abs().max().item()
        return I(torch.cat([
            (-d).clip(0, c),
            d.clip(0, c),
            torch.zeros_like(d),
        ]) / c)
    def pil(self):
        if self.dtype=='pil':
            return self.data
        elif self.dtype=='numpy':
            data = 255*self.data.clip(0,1)
            return Image.fromarray((
                data.transpose(1,2,0) if data.shape[0]!=1
                else data[0]
            ).astype(np.uint8))
        elif self.dtype=='torch':
            return TF.to_pil_image(self.data.float().clamp(0,1))
        assert 0, 'data not understood'
    def numpy(self):
        if self.dtype=='pil':
            return I(np.asarray(self.data)).data
        elif self.dtype=='numpy':
            return self.data
        elif self.dtype=='torch':
            return self.data.numpy()
        assert 0, 'data not understood'
    def torch(self):
        if self.dtype=='pil':
            return TF.to_tensor(self.data)
        elif self.dtype=='numpy':
            return torch.from_numpy(self.data.copy())
        elif self.dtype=='torch':
            return self.data
        assert 0, 'data not understood'
    def uint8(self, ch_last=False):
        ans = (self.np()*255).astype(np.uint8)
        return ans.transpose(1,2,0) if ch_last else ans
    def cv2(self):
        return self.uint8(ch_last=True)[...,::-1]
    def p(self):
        return self.pil()
    def n(self):
        return self.numpy()
    def np(self):
        return self.numpy()
    def t(self):
        return self.torch()
    def tensor(self):
        return self.torch()

    # transforms
    def transpose(self):
        if self.dtype=='pil':
            return I(self.data.transpose(method=Image.TRANSPOSE))
        elif self.dtype=='numpy':
            return I(np.swapaxes(self.data, 1, 2))
        elif self.dtype=='torch':
            return I(self.data.permute(0,2,1))
        assert 0, 'data not understood'
    def T(self):
        return self.transpose()
    def fliph(self):
        if self.dtype=='pil':
            return I(self.data.transpose(method=Image.FLIP_LEFT_RIGHT))
        elif self.dtype=='numpy':
            return I(self.data[...,::-1])
        elif self.dtype=='torch':
            return I(self.data.flip(dims=(2,)))
        assert 0, 'data not understood'
    def flipv(self):
        if self.dtype=='pil':
            return I(self.data.transpose(method=Image.FLIP_TOP_BOTTOM))
        elif self.dtype=='numpy':
            return I(self.data[:,::-1])
        elif self.dtype=='torch':
            return I(self.data.flip(dims=(1,)))
        assert 0, 'data not understood'
    def rotate(self, deg):
        if deg==0:
            return self
        elif deg==90:
            return self.rotate90()
        elif deg==180:
            return self.rotate180()
        elif deg==270:
            return self.rotate270()
        elif deg==360:
            return self
        assert 0, 'data not understood'
    def rotate90(self):
        if self.dtype=='pil':
            return I(self.data.transpose(method=Image.ROTATE_90))
        elif self.dtype in ['numpy', 'torch']:
            return self.transpose().flipv()
    def rotate180(self):
        if self.dtype=='pil':
            return I(self.data.transpose(method=Image.ROTATE_180))
        elif self.dtype in ['numpy', 'torch']:
            return self.fliph().flipv()
    def rotate270(self):
        if self.dtype=='pil':
            return I(self.data.transpose(method=Image.ROTATE_270))
        elif self.dtype in ['numpy', 'torch']:
            return self.transpose().fliph()
    def crop(self, corner_from, size_from=None, size_to=None, size_original=None, resample='nearest', dry=False):
        if dry:
            sf = self.size if size_from is None else size_from
            return Namespace(
                size_original=self.size if size_original is None else size_original,
                corner_from=corner_from,
                size_from=sf,
                size_to=sf if size_to is None else size_to,
            )
        else:
            s = self.crop(
                corner_from=corner_from, size_from=size_from,
                size_to=size_to, size_original=size_original,
                dry=True,
            )
            corner_from = I.sizer(s.corner_from)
            size_from = I.sizer(s.size_from)
            size_to = I.sizer(s.size_to)
            return I(TF.resized_crop(
                self.pil().convert('RGBA'),
                corner_from[0], corner_from[1], size_from[0], size_from[1], size_to,
                interpolation=getattr(TF.InterpolationMode, resample.upper()),
            ))
    def crop_alpha(self, thresh=0.5):
        if not self.mode in {'RGBA', 'L'}:
            return self
        t = self.tensor()
        a = t[-1]
        x = (a.amax(dim=1) > thresh).nonzero()
        y = (a.amax(dim=0) > thresh).nonzero()
        if len(x)==0 or len(y)==0:
            return None
        ans = t[:, x[0]:x[-1]+1][...,y[0]:y[-1]+1]
        return I(ans)

    # resizing
    def resize(self, s, resample='nearest', dry=False):
        if dry:
            return Namespace(
                size_original=self.size,
                corner_from=(0,0),
                size_from=self.size,
                size_to=s,
            )
        if self.dtype=='pil':
            s = I.sizer(self.resize(s, dry=True).size_to)
            return I(self.data.resize(
                s[::-1], resample=getattr(Image, resample.upper()),
            ))
        else:
            return I(self.pil()).resize(s, resample=resample)
    def rescale(self, factor, resample='nearest', dry=False):
        if dry:
            return Namespace(
                size_original=self.size,
                corner_from=(0,0),
                size_from=self.size,
                size_to=tuple(factor*s for s in self.size),
            )
        else:
            return self.resize(
                self.rescale(factor, dry=True).size_to,
                resample=resample,
            )
    def rmax(self, s=512, resample='nearest', dry=False):
        if dry:
            h,w = self.size
            return Namespace(
                size_original=(h,w),
                corner_from=(0,0),
                size_from=(h,w),
                size_to=(s, w*s/h) if w<h else (h*s/w, s),
            )
        else:
            return self.resize(
                self.rmax(s=s, dry=True).size_to,
                resample=resample,
            )
    def rmin(self, s=512, resample='nearest', dry=False):
        if dry:
            h,w = self.size
            return Namespace(
                size_original=(h,w),
                corner_from=(0,0),
                size_from=(h,w),
                size_to=(s, w*s/h) if h<w else (h*s/w, s),
            )
        else:
            return self.resize(
                self.rmin(s=s, dry=True).size_to,
                resample=resample,
            )
    def rdiam(self, s=512, resample='nearest', dry=False):
        return self.rescale(
            factor=s/self.diam,
            resample=resample,
            dry=dry,
        )

    # pixel rounding
    @staticmethod
    def rounder(s, rounding='floor'):
        if rounding=='floor':
            return math.floor(s)
        elif rounding=='round':
            return round(s)
        elif rounding=='ceil':
            return math.ceil(s)
        assert 0, 'rounding not understood'
    @staticmethod
    def sizer(s, rounding='floor'):
        if isinstance(s, Iterable):
            return tuple(I.rounder(q) for q in s)
        elif isinstance(s, float):
            s = I.rounder(s, rounding='floor')
            return (s,s)
        elif isinstance(s, int):
            return (s,s)
        assert 0, 'data not understood'
    @staticmethod
    def tvsize(p, ratio=16/9):
        w = p * ratio
        if int(w)!=w or int(p)!=p:
            warnings.warn(f'non-integer tv size')
        return (int(p), int(w))
    
    # grid
    def left(self, img, bg=0.5):
        return I.grid([img, self], bg=bg)
    def right(self, img, bg=0.5):
        return I.grid([self, img], bg=bg)
    def top(self, img, bg=0.5):
        return I.grid([[img,],[self,]], bg=bg)
    def bottom(self, img, bg=0.5):
        return I.grid([[self,],[img,]], bg=bg)
    @staticmethod
    def grid(imgs, just=True, bg=0.5):
        # repackage
        if isinstance(imgs, torch.Tensor):
            assert len(imgs.shape)==4 and imgs.shape[1] in [1,3,4]
            return I.grid([i for i in imgs], just=just, bg=bg)
        assert isinstance(imgs, list)
        if any(isinstance(i, list) for i in imgs):
            x = [
                [j for j in i] if isinstance(i, list) else [i,]
                for i in imgs
            ]
        else:
            x = [[i for i in imgs],]

        # get sizes
        nrows = len(x)
        ncols = max(len(row) for row in x)
        hs = np.zeros((nrows,ncols))
        ws = np.zeros((nrows,ncols))
        for r in range(nrows):
            row = x[r]
            for c in range(ncols):
                if c==len(row): row.append(None)
                item = row[c]
                if item is None:
                    hs[r,c] = ws[r,c] = 0
                else:
                    item = I(item)
                    hs[r,c], ws[r,c] = item.size
                row[c] = item
        offx = np.cumsum(np.max(hs, axis=1))
        if just:
            offy = np.cumsum(np.max(ws, axis=0))[None,...].repeat(nrows,0)
        else:
            offy = np.cumsum(ws, axis=1)

        # composite
        ans = I.blank((int(offx[-1]), int(offy.max())), c=bg).pil()
        for r in range(nrows):
            for c in range(ncols):
                item = x[r][c]
                if item is not None:
                    ox = offx[r-1] if r>0 else 0
                    oy = offy[r,c-1] if c>0 else 0
                    ans.paste(item.pil().convert('RGBA'), (int(oy),int(ox)))
        return I(ans)

    # color
    _COLOR_DICT = {
        'r': (1,0,0),
        'g': (0,1,0),
        'b': (0,0,1),
        'k': 0,
        'w': 1,
        't': (0,1,1),
        'm': (1,0,1),
        'y': (1,1,0),
        'a': (0,0,0,0),
    }
    @staticmethod
    def c255(c, c255=True):
        # color format utility
        # returns RGBA as uint tuple
        if c is None:
            return None
        if isinstance(c, str):
            c = I._COLOR_DICT[c]
        if isinstance(c, list) or isinstance(c, tuple):
            if len(c)==3:
                c = c + (1,)
            elif len(c)==1:
                c = (c,c,c,1)
            c = tuple(int(255*q) for q in c)
        else:
            c = int(255*c)
            c = (c,c,c,255)
        if not c255:
            c = tuple(ch/255.0 for ch in c)
        return c
    @staticmethod
    def uniform_colors(n, seed=None, lightness=0.5, saturation=0.5, roll=0, c255=False):
        if seed is not None:
            with uutil.numpy_seed(seed):
                perm = np.random.permutation(n)
        else:
            perm = np.arange(n)
        return [
            I.c255(colorsys.hls_to_rgb(hue, lightness, saturation), c255=c255)
            for hue in np.modf(np.linspace(0, 1, n, endpoint=False)+roll)[0][perm]
        ]

    # compositing
    @staticmethod
    def blank(size, c='k', rounding='floor'):
        size = I.sizer(size, rounding=rounding)
        c = I.c255(c)
        return I(
            np.ones((4,*size)) * np.asarray(c)[:,None,None]/255.0
        )
    def channel_set(self, ch, val=0.0):
        d = self.torch().clone()
        d[ch] = val
        return I(d)
    def alpha_set(self, alpha):
        alpha = I(alpha)
        assert alpha.mode=='L'
        alpha = alpha.np()
        rgb = (self.convert('RGB') if self.mode=='L' else self).np()[:3]
        return I(np.concatenate([rgb, alpha]))
    def alpha_composite(self, img, opacity=1):
        if img is None or opacity==0:
            return I(self)
        img = I(img)
        assert img.mode=='RGBA'
        if opacity!=1:
            img = img.np().copy()
            img[3] *= opacity
            img = I(img)
        return I(Image.alpha_composite(self.convert('RGBA').pil(), img.pil()))
    def acomp(self, *args, **kwargs):
        return self.alpha_composite(*args, **kwargs)
    def background_composite(self, img, opacity=1):
        return I(img).alpha_composite(self, opacity=opacity)
    def bcomp(self, *args, **kwargs):
        return self.background_composite(*args, **kwargs)
    def background(self, c=0.5):
        return I.blank(self.size, c=c).alpha_composite(self)
    def bg(self, *args, **kwargs):
        return self.background(*args, **kwargs)
    def l2rgba(self, c='k'):
        # turns bw to solid color, using L as alpha
        assert self.mode=='L'
        return I.blank(self.size, c=c).alpha_set(self.np())
    @staticmethod
    def average(imgs):
        imgs = [I(i) for i in imgs]
        assert len(set([i.mode for i in imgs]))==1
        return I(np.mean(np.stack([i.np() for i in imgs]), axis=0))
    @staticmethod
    def avg(*args, **kwargs):
        return I.average(*args, **kwargs)
    @staticmethod
    def mean(*args, **kwargs):
        return I.average(*args, **kwargs)

    # drawing
    def line(self, a, b, w=1, c='r'):
        a = I.sizer(a, rounding='floor')
        b = I.sizer(b, rounding='floor')
        c = I.c255(c)
        w = max(1, math.floor(w))
        ans = self.convert('RGBA').pil().copy()
        d = PIL.ImageDraw.Draw(ans)
        d.line([a[::-1], b[::-1]], fill=c, width=w)
        return I(ans)
    def point(self, xy, s=1, c='r'):
        c = I.c255(c)
        x,y = I.sizer(xy, rounding='floor')
        ans = self.convert('RGBA')
        if s==0:
            ans = ans.uint8()
            ans[:,x,y] = c
            ans = I(ans)
        else:
            ans = ans.pil().copy()
            d = PIL.ImageDraw.Draw(ans)
            d.ellipse(
                [(y-s,x-s), (y+s,x+s)],
                fill=c,
            )
        return I(ans)
    def box(self, corner, size, w=1, c='r', f=None):
        corner = I.sizer(corner, rounding='floor')
        size = I.sizer(size, rounding='floor')
        w = max(1, math.floor(w))
        c = I.c255(c)
        f = I.c255(f)
        ans = self.convert('RGBA').pil().copy()
        d = PIL.ImageDraw.Draw(ans)
        d.rectangle(
            [corner[1], corner[0], corner[1]+size[1]-1, corner[0]+size[0]-1],
            fill=f, outline=c, width=w,
        )
        return I(ans)
    def border(self, w=1, c='r', pad=False):
        i,j = self.size
        ans = self if not pad else self.crop((-w,-w), (i+2*w,j+2*w))
        return ans.box(
            (0, 0),
            ans.size,
            w=w, c=c, f=None,
        )
    def borderp(self, w=1, c=0.5, pad=True):
        return self.border(w=w, c=c, pad=pad)

    # text
    def annotate(self, text, pos, s=12, anchor='tl', c='m', bg='k', spacing=None, padding=0):
        t = I.text(
            text, s=s, c=c, bg=bg,
            spacing=spacing, padding=padding,
        )
        x,y = pos
        x = {
            't': x,
            'b': x-t.size[0],
            'c': x-t.size[0]/2,
        }[anchor[0].lower()]
        y = {
            'l': y,
            'r': y-t.size[1],
            'c': y-t.size[1]/2,
        }[anchor[1].lower()]
        t = t.convert('RGBA').pil()
        ans = self.convert('RGBA').pil().copy()
        ans.paste(t, I.sizer((y,x), rounding='floor'), t)
        return I(ans)
    def annot(self, *args, **kwargs):
        return self.annotate(*args, **kwargs)
    def caption(self, text, s=24, pos='t', c='w', bg='k', spacing=None, padding=None):
        pos = pos[0].lower()
        t = I.text(text, s=s, c=c, bg=bg, spacing=spacing, padding=padding)
        if pos=='t':
            return self.top(t, bg=bg)
        elif pos=='b':
            return self.bottom(t, bg=bg)
        elif pos=='l':
            return self.left(t, bg=bg)
        elif pos=='r':
            return self.right(t, bg=bg)
        assert 0, 'data not understood'
    def cap(self, *args, **kwargs):
        return self.caption(*args, **kwargs)
    @staticmethod
    def text(
                text,
                s=24,
                facing='right',  # write in this direction
                pos='left',      # align to this position
                c='w',
                bg='k',
                h=None,
                w=None,
                spacing=None,  # between lines
                padding=None,  # around entire thing
                force_size=False,
            ):
        # text image utility
        text = str(text)
        s = max(1, math.floor(s))
        spacing = math.ceil(s*4/10) if spacing is None else spacing
        padding = math.ceil(s*4/10) if padding is None else padding
        facing = facing.lower()
        if facing in ['u', 'up', 'd', 'down']:
            h,w = w,h
        c,bg = I.c255(c), I.c255(bg)
        f = PIL.ImageFont.truetype(_FN_ARIAL, s)

        td = PIL.ImageDraw.Draw(Image.new('RGBA', (1,1), (0,0,0,0)))
        tw,th = td.multiline_textsize(text, font=f, spacing=spacing)
        if not force_size:
            if h and h<th: h = th
            if w and w<tw: w = tw
        h = h or th+2*padding
        w = w or tw+2*padding

        pos = pos.lower()
        an = None
        if pos in ['c', 'center']:
            xy = (w//2, h//2)
            an = 'mm'
            align = 'center'
        elif pos in ['l', 'lc', 'cl', 'left']:
            xy = (padding, h//2)
            an = 'lm'
            align = 'left'
        elif pos in ['r', 'rc', 'cr', 'right']:
            xy = (w-padding, h//2)
            an = 'rm'
            align = 'right'
        elif pos in ['t', 'tc', 'ct', 'top']:
            xy = (w//2, padding)
            an = 'ma'
            align = 'center'
        elif pos in ['b', 'bc', 'cb', 'bottom']:
            xy = (w//2, h-padding)
            an = 'md'
            align = 'center'
        elif pos in ['tl', 'lt']:
            xy = (padding, padding)
            align = 'left'
        elif pos in ['bl', 'lb']:
            xy = (padding, h-padding-th)
            align = 'left'
        elif pos in ['tr', 'rt']:
            xy = (w-padding-tw, padding)
            align = 'right'
        elif pos in ['br', 'rb']:
            xy = (w-padding-tw, h-padding-th)
            align = 'right'
        else:
            assert False, 'pos not understood'
        
        ans = Image.new('RGBA', (w,h), bg)
        d = PIL.ImageDraw.Draw(ans)
        d.multiline_text(
            xy, text, fill=c, font=f, anchor=an,
            spacing=spacing, align=align,
        )

        if facing in ['l', 'left']:
            ans = ans.rotate(180)
        elif facing in ['u', 'up']:
            ans = ans.rotate(90, expand=True)
        elif facing in ['d', 'down']:
            ans = ans.rotate(-90, expand=True)
        return I(ans)
    
    # ipython integration
    def _repr_png_(self):
        bio = io.BytesIO()
        self.pil().save(bio, 'PNG')
        return bio.getvalue()

    # saving
    def save(self, fn, count=False):
        if count:
            cnt = 0
            q = fnstrip(fn,1)
            fn = f'{q.dn}/{q.bn}-{cnt:04d}.{q.ext}'
            while os.path.isfile(fn):
                cnt += 1
                fn = f'{q.dn}/{q.bn}-{cnt:04d}.{q.ext}'
        if fn.lower().endswith('.exr'):
            if self.mode in ['RGB', 'L']:
                cv2.imwrite(fn, self.np().transpose(1,2,0)[...,::-1].astype(np.float32))
            elif self.mode=='RGBA':
                cv2.imwrite(fn, self.np().transpose(1,2,0)[...,[2,1,0,3]].astype(np.float32))
            else:
                assert 0
        else:
            self.pil().save(fn)
        return fn
    def uri(self):
        bio = io.BytesIO()
        self.pil().save(bio, 'PNG')
        encoded = base64.b64encode(bio.getvalue())
        return 'data:image/png;base64,'+encoded.decode('utf-8')

    # functional
    def apply(self, func, *args, **kwargs):
        return func(self, *args, **kwargs)
    def f(self, func, *args, **kwargs):
        return self.apply(func, *args, **kwargs)


