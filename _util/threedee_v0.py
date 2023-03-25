


from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch


try:
    import igl
except:
    pass

try:
    import meshplot as mp # https://skoch9.github.io/meshplot/tutorial/
except:
    pass

try:
    import pygltflib
except:
    pass

try:
    import moderngl
    import moderngl_window as mglw
except:
    pass

try:
    import pyrr
    from pyrr import Matrix44
except:
    pass


# pygltflib.COMPONENT_TYPES
# https://github.com/KhronosGroup/glTF/blob/master/specification/2.0/README.md#accessors
_component_dtypes = {
    5120: np.int8,    # 1 byte
    5121: np.uint8,   # 1 unsigned_byte
    5122: np.int16,   # 2 short
    5123: np.uint16,  # 2 unsigned_short
    5125: np.uint32,  # 4 unsigned_int
    5126: np.float32, # 4 float
}
_accessor_ncomps = {
    'SCALAR': (1,),
    'VEC2':   (2,),
    'VEC3':   (3,),
    'VEC4':   (4,),
    'MAT2':   (2,2), # 4
    'MAT3':   (3,3), # 9
    'MAT4':   (4,4), # 16
}
_mesh_primitive_modes = {
    0: 'POINTS',
    1: 'LINES',
    2: 'LINE_LOOP',
    3: 'LINE_STRIP',
    4: 'TRIANGLES',
    5: 'TRIANGLE_STRIP',
    6: 'TRIANGLE_FAN',
}


###### mesh utils ######

def tohomo(x):
    return np.concatenate([
        x, np.ones((len(x),1)),
    ], axis=1)
def dehomo(x):
    return x[:,:-1]

def read_obj(fn):
    o = read(fn).split('\n')
    v = []
    f = []
    for line in [line.split(' ') for line in o]:
        if line[0]=='v':
            v.append([float(i) for i in line[1:]])
        if line[0]=='f':
            f.append([int(i)-1 for i in line[1:]])
    return np.asarray(v), np.asarray(f, dtype=int)
def face_subset(f, wh):
    wv = wh
    faces = f
    wf = np.isin(faces, np.where(wv)[0]).all(axis=1)
    faces = (np.cumsum(wv)-1)[faces[wf]]
    return faces
def point_corresp(a, b, batch=50000*1000, device='cuda', bar=False):
    na,nb = len(a), len(b)
    # a,b = a.astype(np.double), b.astype(np.double)
    n = batch//nb  # number of A to do per batch
    a,b = torch.tensor(a).to(device), torch.tensor(b).to(device)
    idx = []
    dist = []
    for ia in (tqdm if bar else lambda x: x)(chunk_rows(a, na//n+1)):
        # d = igl.all_pairs_distances(
        #     ia, b, False,
        # )
        d = torch.norm(
            (ia[:,None,:] - b[None,:,:]),
            dim=-1,
        )
        i = torch.argmin(d, dim=1)
        idx.append(i)
        dist.append(d[np.arange(len(ia)), i])
    return torch.cat(idx).cpu().numpy(), torch.cat(dist).cpu().numpy()

def bary2cart(mesh, bary_v, bary_f):
    # bary_v = (n,abc) in simplex
    # bary_f = (n,) list of faces
    if type(mesh)==tuple:
        # vertex-based (i.e. vert location)
        # mesh = v(V,D), f(F,abc)
        v,f = mesh
        return np.sum(
            v[f[bary_f]] * bary_v[...,None],
            axis=1,
        )
    else:
        # face-based (i.e. textures)
        # mesh = (F,abc,D)
        return np.sum(
            mesh[bary_f] * bary_v[...,None],
            axis=1,
        )
def cart2bary(mesh, cart_v, cart_f):
    v,f = mesh
    t = v[f[cart_f]]
    a,b,c = t[:,0],t[:,1],t[:,2]
    v0 = b - a
    v1 = c - a
    v2 = cart_v - a
    d00 = np.sum(v0 * v0, axis=1)
    d01 = np.sum(v0 * v1, axis=1)
    d11 = np.sum(v1 * v1, axis=1)
    d20 = np.sum(v2 * v0, axis=1)
    d21 = np.sum(v2 * v1, axis=1)
    denom = d00 * d11 - d01 * d01
    denom[denom==0.0] = 1.0  # hack against no area
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return np.stack([u,v,w]).T
def remove_innards(verts, faces, norms, uvcol, uvmap, texidxs, n=1, thresh=1.3):
    # high thresh ==> keep more
    wind = igl.fast_winding_number_for_meshes(
        v=verts, f=faces.astype(np.int32), q=verts,
    )
    wv = wind < thresh
    wf = np.isin(faces, np.where(wv)[0]).all(axis=1)
    faces = (np.cumsum(wv)-1)[faces[wf]]
    ans = (
        verts[wv],
        faces,
        norms[wv],
        uvcol[wv],
        uvmap[wv],
        texidxs[wv],
    )
    if n==1:
        return ans
    else:
        return remove_innards(
            *ans,
            n=n-1,
            thresh=thresh,
        )


###### vis utils ######

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


###### gltf utils ######

# gltf helpers
def gltf_accessor(gltf, accessor_idx):
    acc = gltf.accessors[accessor_idx]
    bv = gltf.bufferViews[acc.bufferView]
    blob = gltf.binary_blob()
    # blob = bins[bv.buffer] if bins else gltf.load_file_uri(gltf.buffers[bv.buffer].uri)
    arr = np.frombuffer(
        blob,
        dtype=u3d._component_dtypes[acc.componentType],
        count=acc.count*np.prod(u3d._accessor_ncomps[acc.type]),
        offset=bv.byteOffset+acc.byteOffset,
    ).reshape(acc.count, *u3d._accessor_ncomps[acc.type])
    return arr
def gltf_image(gltf, image_idx):
    img = gltf.images[image_idx]
    bv = gltf.bufferViews[img.bufferView]
    blob = gltf.binary_blob()
    ans = blob[bv.byteOffset:bv.byteOffset+bv.byteLength]
    ans = I(Image.open(io.BytesIO(ans)))
    return ans
def sample_texture(bary, face_idxs, faces, uvs, texture, texture_idxs, base_colors, return_more=False):
    # prep textures
    # tex = np.stack([
    #     np.moveaxis(np.asarray(
    #         t.pil().rotate(-90).transpose(Image.FLIP_LEFT_RIGHT)
    #     ), 2, 0)
    #     for t in texture
    # ])
    tex = texture

    # sample textures (truncated-uv)
    tuvcl = bary2cart((uvs,faces), bary, face_idxs)
    tuvcl = tuvcl - np.floor(tuvcl)
    tucl = (tuvcl[:,0]*(tex.shape[-1]-1)+0.5).astype(int)
    tvcl = (tuvcl[:,1]*(tex.shape[-1]-1)+0.5).astype(int)
    tmcl = texture_idxs[faces[face_idxs,0]]
    trgba = tex[tmcl, :, tucl, tvcl] / 255.0 * base_colors[tmcl]

    if not return_more:
        return trgba
    else:
        return trgba, {
            'trgba': trgba,
            'tex': tex,
            'tmuv': np.stack([
                tmcl, tucl, tvcl,
            ], axis=0).T,
        }
def sample_texture_uv(tex, uv):
    uv = uv-np.floor(uv)
    uv = np.round(uv*(np.asarray(tex.size[::-1])-1)[None,]).astype(np.int32)
    return tex.numpy()[:,uv[:,1],uv[:,0]].T



