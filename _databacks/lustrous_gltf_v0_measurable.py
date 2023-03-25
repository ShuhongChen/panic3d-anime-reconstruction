



from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
from _util.twodee_v1 import * ; import _util.twodee_v1 as u2d
from _util.threedee_v0 import * ; import _util.threedee_v0 as u3d



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
# exec(read('./hack/util/gltf_v0.py'))
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


# mesh utils
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
# def remove_innards(verts, faces, norms, uvcol, uvmap, texidxs, n=1, thresh=1.3):
def remove_innards(verts, faces, n=1, thresh=1.3):
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
        # norms[wv],
        # uvcol[wv],
        # uvmap[wv],
        # texidxs[wv],
    )
    if n==1:
        return ans
    else:
        return remove_innards(
            *ans,
            n=n-1,
            thresh=thresh,
        )
def get_head_bone(gltf):
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

    # get interesting nodes
    inodes = Dict()
    hbones = gltf.extensions['VRM']['humanoid']['humanBones']
    for hb in hbones:
        if hb['bone']=='neck':
            inodes['neck'] = hb['node']
        elif hb['bone']=='head':
            inodes['head'] = hb['node']
        elif hb['bone']=='leftEye':
            inodes['eye_left'] = hb['node']
        elif hb['bone']=='rightEye':
            inodes['eye_right'] = hb['node']
    assert 'head' in inodes.keys()
    # assert 'neck' in inodes.keys() and 'head' in inodes.keys()
    # assert set(inodes.keys())=={'neck', 'head', 'eye_left', 'eye_right'}

    g_skin = gltf.skins[0]
    ibms = np.transpose(gltf_accessor(gltf, accessor_idx=g_skin.inverseBindMatrices), (0,2,1))
    # ibm_neck = ibms[g_skin.joints.index(inodes['neck'])]
    ibm_head = ibms[g_skin.joints.index(inodes['head'])]
    # loc_bone_neck = -ibm_neck[:3,-1]
    loc_bone_head = -ibm_head[:3,-1]
    # return loc_bone_neck, loc_bone_head
    return loc_bone_head




# class container
class LustrousGLTF:
    # load gltf
    def __init__(self, fn):
        # load gltf
        self.fn = fn
        self.gltf = gltf = pygltflib.GLTF2().load_binary(self.fn)
        
        # get combined attributes
        _verts = []
        _norms = []
        _faces = []
        _uvcol = []
        _uvmap = []
        _texidxs = []
        _basecol = []
        vc = 0
        tc = 0
        timgs = []
        for mesh in gltf.meshes:
            for prim in mesh.primitives:
                # for triangle meshes
                assert u3d._mesh_primitive_modes[prim.mode]=='TRIANGLES'

                attributes = {
                    k: gltf_accessor(gltf, v)
                    for k,v in json.loads(prim.attributes.to_json()).items()
                    if v is not None
                }
                indices = gltf_accessor(gltf, prim.indices)

                # material = gltf.materials[prim.material]
                # texture = gltf.textures[material.pbrMetallicRoughness.baseColorTexture.index]
                # texture_img = gltf_image(gltf, texture.source)
                # texture_set = material.pbrMetallicRoughness.baseColorTexture.texCoord
                # timgs.append(texture_img)
                # try:
                #     bc = material.pbrMetallicRoughness.baseColorFactor
                # except:
                #     bc = [1,1,1,1]

                verts = attributes['POSITION']
                # norms = attributes['NORMAL']
                faces = indices.reshape(-1, 3) + vc
                # uvmap = attributes[f'TEXCOORD_{texture_set}']
                # uvcol = sample_texture_uv(texture_img, uvmap)
                _verts.append(verts)
                # _norms.append(norms)
                _faces.append(faces)
                # _uvcol.append(uvcol)
                # _uvmap.append(uvmap)
                # _texidxs.append(tc * np.ones(len(verts), dtype=int))
                # _basecol.append(bc)
                vc += len(verts)
                # tc += 1
        self.verts = np.concatenate(_verts)
        self.faces = np.concatenate(_faces)
        # self.normals = np.concatenate(_norms)
        # self.uv_colors = np.concatenate([i[:,:3] for i in _uvcol])
        # self.uv_map = np.concatenate(_uvmap)
        # self.texture_idxs = np.concatenate(_texidxs)
        # self.textures = timgs
        # self.base_colors = np.asarray(_basecol)
        return
    
    # mesh adjustment
    def remove_innards(self, n=1, thresh=1.3):
        (
            self.verts,
            self.faces,
            # self.normals,
            # self.uv_colors,
            # self.uv_map,
            # self.texture_idxs,
        ) = remove_innards(
            self.verts,
            self.faces,
            # self.normals,
            # self.uv_colors,
            # self.uv_map,
            # self.texture_idxs,
            n=n, thresh=thresh,
        )
        return self
    
class LustrousGLTFDecapitated:
    def __init__(self, gltf_lustrous, offset_head=[0, 0.1, 0], boxwarp=0.5, texture_cache_size=1024):
        self.gltf_lustrous = gltf_lustrous
        self.offset_head = np.asarray(offset_head)
        self.boxwarp = boxwarp
        
        # get head location
        self.loc_bone_head = get_head_bone(self.gltf_lustrous.gltf)
        self.loc_origin = self.loc_bone_head + self.offset_head
        
        # decapitate (recenter verts, and remove disqualifying faces only)
        verts = self.gltf_lustrous.verts
        verts = verts - self.loc_origin[None,]
        vkeep = (np.abs(verts) <= self.boxwarp/2).all(axis=1)
        fkeep = vkeep[self.gltf_lustrous.faces].all(axis=1)
        self.faces = self.gltf_lustrous.faces[fkeep].astype(np.int64)
        self.verts = verts
        
        # # cache textures
        # self.texture_cache_size = tcs = texture_cache_size
        # self.textures = np.stack([
        #     np.moveaxis(np.asarray(
        #         t.resize(tcs,'bilinear').pil().rotate(-90).transpose(Image.FLIP_LEFT_RIGHT)
        #     ), 2, 0)
        #     for t in self.gltf_lustrous.textures
        # ])
        return
    
    def sample_points_uniform(self, n_sample, seed=None):
        q = np.random.RandomState(seed).rand(n_sample, 3)
        q = (q - 0.5) * self.boxwarp
        return q
    def sample_points_near_surface(self, n_sample, sigma, seed=None, clip=True):
        n = n_sample
        v = self.verts
        f = self.faces
        
        # sample points near mesh
        with np_seed(seed):
            qmeshb,qmeshf = igl.random_points_on_mesh(n, v, f)
            purt = sigma * np.random.randn(*qmeshb.shape)
        query_mesh = bary2cart(
            (v,f), qmeshb, qmeshf,
        ) + purt
        if clip:
            hbw = self.boxwarp / 2
            query_mesh = np.clip(query_mesh, -hbw, hbw)
        return query_mesh

    def get_point_distance(self, queries):
        query_mesh = queries
        n = len(queries)
        v = self.verts
        f = self.faces
        
        # find closest point on mesh
        dist2,fcl,vcl = igl.point_mesh_squared_distance(
            query_mesh, v, f,
        )
        dist = np.sqrt(dist2)[...,None]
        return dist
    def get_point_colors(self, queries):
        query_mesh = queries
        n = len(queries)
        v = self.verts
        f = self.faces
        
        # find closest point on mesh
        dist2,fcl,vcl = igl.point_mesh_squared_distance(
            query_mesh, v, f,
        )
        dist = np.sqrt(dist2)[...,None]
        bcl = cart2bary(
            (v,f), vcl, fcl,
        )
        # vcl_ = util_graphics.bary2cart((mhv,mhf), bcl, fcl) # == vcl
        
        # get colors
        cs = sample_texture(
            bcl, fcl, f,
            self.gltf_lustrous.uv_map,
            self.textures, self.gltf_lustrous.texture_idxs, self.gltf_lustrous.base_colors,
        )
        return cs
    def get_point_winding(self, queries):
        q = queries
        v = self.verts
        wind = igl.fast_winding_number_for_meshes(
            v=v,
            f=self.gltf_lustrous.faces.astype(np.int32),
            q=q.astype(v.dtype).copy(),
        )
        return wind
    
    # combined sampling
    def sample(self, n_uniform, n_surface, sigma_surface=0.01, seed=None, rm_alpha=True):
        head = self
        queries = np.concatenate([
            head.sample_points_uniform(n_uniform, seed=seed),
            head.sample_points_near_surface(n_surface, sigma_surface, seed=seed),
        ])
        dist = head.get_point_distance(queries)
        colors = head.get_point_colors(queries)
        wind = head.get_point_winding(queries)
        proto = np.concatenate([np.zeros(n_uniform), np.ones(n_surface)])

        # filter out alpha-colored points
        if rm_alpha:
            c = colors[:,-1]>0.5
            queries = queries[c]
            dist = dist[c]
            colors = colors[c]
            wind = wind[c]
            proto = proto[c]
        return Dict({
            'xyz': queries,
            'distances': dist,
            'colors': colors,
            'winding': wind,
            'sampling_protocol': proto,
        })

# gltf = LustrousGLTF(f'./_data/lustrous/raw/dssc/{char_bn}.vrm').remove_innards()
# head = LustrousGLTFDecapitated(gltf)





