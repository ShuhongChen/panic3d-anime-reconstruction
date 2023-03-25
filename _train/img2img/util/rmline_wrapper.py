




from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
from _util.twodee_v1 import * ; import _util.twodee_v1 as u2d

import _util.serving_v1 as userving
from _util import sketchers_v2 as usketch



class RMLineWrapper(nn.Module):
    def __init__(self, inferquery):
        super().__init__()
        self.inferquery = inferquery
        ckpt = userving.Checkpoint(*inferquery)
        self.model = ckpt.model()#.to(device)
        return
    def forward(self, img, kpts):
        device = self.model.generator[0].weight.device
        img_alpha = img['a']
        img = img.bg('w').convert('RGB').t()[None].to(device)
        fhull = facehull(I(img), kpts).to(device)
        dog = usketch.batch_dog(
            img,
            t=1.0,
            sigma=0.5,
            k=1.6,
            epsilon=0.01,
            kernel_factor=4,
        )>0.5
        d = 2
        if d>1:
            dog = kornia.morphology.dilation(
                dog.float(),
                torch.ones(d,d).to(dog.device),
            ).bool()
        dog = (dog & ~fhull.bool()).float()
        xin = {
            'image': I(img).convert('RGBA').bg('w').convert('RGB').t()[None,].to(device),
            'face_hull': fhull.to(device),
            'line_mask': dog.to(device),
        }
        out = self.model.forward(xin)
        ans = torch.lerp(xin['image'], out['image'], xin['line_mask'])
        ans = I(ans).alpha_set(img_alpha)
        return ans


import requests
def anime_face_detector_api(img):
    resp = requests.get(
        url='http://localhost:5000/detectorapi',
        json={
            'image': img.uri(),
        },
    )
    ans = resp.json()
    ans['bbox'] = np.asarray(ans['bbox'])
    ans['keypoints'] = np.asarray(ans['keypoints'])
    return ans
keypoint_groups = Dict(
    chin = [0,1,2,3,4],
    eyelash_right = [5,6,7],
    eyelash_left = [8,9,10],
    eye_right = [
        11,
        12,
        13,
        14,
        15,
        16,
    ],
    eye_left = [
        17,
        18,
        19,
        20,
        21,
        22,
    ],
    nose = [23],
    mouth = [24,25,26,27],
)
def facehull(img, kpts, dilate=5):
    mkpts = kpts
    vr = torch.zeros(1,*img.shape[-2:])
    vl = torch.zeros(1,*img.shape[-2:])
    vm = torch.zeros(1,*img.shape[-2:])
    vn = torch.zeros(1,*img.shape[-2:])
    # v = I.blank(img.shape[-1])
    for a,b in mkpts[keypoint_groups.eye_right].astype(int):
        vr[:,a,b] = 1
    vr = skimage.morphology.convex_hull_image(vr.numpy()[0])[None]
    for a,b in mkpts[keypoint_groups.eye_left].astype(int):
        vl[:,a,b] = 1
    vl = skimage.morphology.convex_hull_image(vl.numpy()[0])[None]
    for a,b in mkpts[keypoint_groups.mouth].astype(int):
        vm[:,a,b] = 1
    vm = skimage.morphology.convex_hull_image(vm.numpy()[0])[None]
    a,b = mkpts[keypoint_groups['nose'][0]].astype(int)
    vn[:,a,b] = 1
    vn = vn.numpy()
    v = I((vr+vl+vm+vn).astype(bool))

    for grp in ['eyelash_left', 'eyelash_right',]:
        g = mkpts[keypoint_groups[grp]]
        for a,b in zip(g[:-1], g[1:]):
            v = v.line(a, b, c='w')
    # g = mkpts[keypoint_groups.mouth]
    # for a,b in zip(g, np.roll(g,1,axis=0)):
    #     v = v.line(a, b, c='w')
    v = kornia.morphology.dilation(
        v.t()[None,:1],
        torch.ones(dilate,dilate),
    )[0]
    return v
def _apply_M_keypoints(M, kpts):
    kpts = kpts[0]
    scores = kpts[:,2:]
    return np.concatenate([
        (M @ np.concatenate([
            kpts[:,:2], np.ones((kpts.shape[0],1))
        ], axis=-1).T).T[:,:2],
        scores,
    ], axis=-1)[None,]
def quickvis(img, preds, thresh=0.5, size=0.001, c='r'):
    v = img
    ms = min(img.size)
    for i in range(preds['n_detections']):
        bbox,kpts = preds.bbox[i], preds.keypoints[i]
        bx,by,bh,bw,bs = bbox
        if bs<thresh: continue
        v = v.box((bx,by), (bh,bw), w=size*ms).annot(f'{bs:.4f}', (bx,by), s=50*size*ms)
        for kx,ky,_ in kpts:
            v = v.point((kx,ky), s=5*size*ms, c=c)
    return v







