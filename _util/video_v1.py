


from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
from _util.twodee_v1 import * ; import _util.twodee_v1 as u2d

try:
    from nvidia import dali
    from nvidia.dali.plugin import pytorch as _
except:
    pass


############################## READERS ##############################

def video_timestamp(frame, fps=24):
    # converts frame number to timestamp string
    f = frame % fps
    s = int(frame / fps) % 60
    m = int(frame / fps / 60)
    return f'{m:03d}:{s:02d}+{f:02d}'
def video_metadata(fn, cap=None):
    # uses existing cap if possible
    assert os.path.isfile(fn)
    release = cap is None
    if release: cap = cv2.VideoCapture(fn)
        
    # frame count + fps
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if int(fps)!=fps:
        warnings.warn(f'fps={fps} not integer')
    if int(frame_count)!=frame_count:
        warnings.warn(f'frame_count={frame_count} not integer')
    fps = int(fps)
    frame_count = int(frame_count)
    
    # size
    size = (
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        cap.get(cv2.CAP_PROP_FRAME_WIDTH),
    )
    if any([int(s)!=s for s in size]):
        warnings.warn(f'size={size} not integer')
    size = tuple(int(s) for s in size)
    shape = (
        frame_count,
        3,  # assume rgb...
        *size,
    )

    # return
    if release: cap.release()
    return {
        'frame_count': frame_count,
        'fps': fps,
        'size': size,
        'shape': shape,
    }

class VideoReaderCV2:
    def __init__(self, fn):
        self.fn = fn
        assert os.path.isfile(self.fn), f'video file {self.fn} not found'
        self.cap = cv2.VideoCapture(self.fn)
        for k,v in video_metadata(self.fn, cap=self.cap).items():
            self.__setattr__(k, v)
        return
    def release(self):
        return self.cap.release()
    
    def timestamp(self, frame):
        f = frame % self.fps
        s = int(frame / self.fps) % 60
        m = int(frame / self.fps / 60)
        return f'{m:03d}:{s:02d}+{f:02d}'
    def seconds(self, frame):
        return frame / self.fps
    def frame(self, s=0, m=0, h=0, f=0):
        return int((s + 60*m + 60*60*h) * self.fps + f)
    def __len__(self):
        return self.frame_count
    def __getitem__(self, idx):
        # acts just like an np.ndarray
        if isinstance(idx, int) or np.issubdtype(type(idx), np.integer):
            if idx<0: idx = len(self)+idx
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            _,frame = self.cap.read()
            ans = I(frame[...,::-1])
            return ans.np()
        elif isinstance(idx, list):
            return np.stack([self[i] for i in idx])
        elif isinstance(idx, slice) or isinstance(idx, range):
            a,b,c = idx.start, idx.stop, idx.step
            if a is None: a = 0
            if b is None: b = len(self)
            if c is None: c = 1
            if a<0: a = len(self)+a
            if b<0: b = len(self)+b
            idx = range(a,b,c)
            return np.stack([self[i] for i in idx])
        elif isinstance(idx, tuple):
            rest = ((slice(None),),())[isinstance(idx[0],int)] + idx[1:]
            return self[idx[0]][rest]
        else:
            assert 0, f'idx={idx} not understood'


############################## WRITERS ##############################

def write_animation(imgs, fn, **kwargs):
    if fn.lower().endswith('.gif'):
        return write_gif(imgs, fn, **kwargs)
    elif fn.lower().endswith('.webp'):
        return write_webp(imgs, fn, **kwargs)
    else:
        assert f'extension {fstrip(fn,1).ext} not understood'
write_ani = write_animation
def write_gif(imgs, fn, fps=1, loop=0, disposal=1):
    assert fn.lower().endswith('.gif')
    imgs = [i.pil() for i in imgs]
    dur = 1000/fps if not isinstance(fps, list) \
        else [1000/f for f in fps]
    return imgs[0].save(
        fn,
        format='GIF',
        append_images=imgs[1:],
        save_all=True,
        include_color_table=True,
        interlace=True,
        optimize=True,
        duration=dur,
        # fps
        loop=loop,
        # num times to loop, 0 forever
        disposal=disposal,
        # 0 no spec
        # 1 don't dispose
        # 2 restore to bg color
        # 3 restore to prev content
    )
def write_webp(
        imgs, fn, fps=1, loop=0,
        lossless=True, bg='k',
        quality=80, method=4,
        minimize_size=False, allow_mixed=False,
    ):
    assert fn.lower().endswith('.webp')
    imgs = [i.pil() for i in imgs]
    dur = 1000/fps if not isinstance(fps, list) \
        else [1000/f for f in fps]
    return imgs[0].save(
        fn,
        append_images=imgs[1:],
        save_all=True,
        duration=int(dur),
        loop=loop,     # 0 forever
        lossless=lossless,
        quality=quality,  # 1-100
        method=method,  # 0-6 (fast-better)
        background=I.c255(bg),
        minimize_size=minimize_size,  # slow write
        allow_mixed=allow_mixed,  # mixed compression
    )
def copy_video_audio(source, target, postfix='_audio'):
    fs = fstrip(target, return_more=True)
    ofn = f'{fs["dn"]}/{fs["bn"]}{postfix}.{fs["ext"]}'
    sp = subprocess.run([
        '/usr/bin/ffmpeg',
        '-i', target,
        '-i', source,
        '-c:v', 'copy',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-c:a', 'aac',
        '-b:a', '192k',
        ofn,
    ])
    assert sp.returncode==0
    return ofn

class VideoWriterCV2:
    def __init__(self, fn, fps=24, overwrite=True):
        self.fn = fn
        self.fps = fps
        self.overwrite = overwrite
        self.initialized = False
        self.size = None
        self.writer = None
        self.frame_count = 0
        return
    def write(self, frame):
        # initialize if not already
        if not self.initialized:
            if os.path.isfile(self.fn):
                if self.overwrite:
                    os.remove(self.fn)
                else:
                    assert 0, f'{self.fn} already exists'
            self.size = frame.size
            self.writer = cv2.VideoWriter(
                self.fn,
                cv2.VideoWriter_fourcc(*'MP4V'),
                self.fps,
                self.size[::-1],
            )
            self.initialized = True
            
        # write and update
        # for some reason, must convert to rgb before cv2
        if frame.size!=self.size:
            warnings.warn(f'frame.size={frame.size} != self.size={self.size}')
        self.writer.write(frame.convert('RGB').cv2())
        self.frame_count += 1
        return
    def release(self):
        self.writer.release()
        return







