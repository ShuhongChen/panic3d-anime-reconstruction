


import os, sys
if '__INIT_NOTEBOOK__' not in globals():
    assert os.getcwd().split('/')[-1]=='_notebooks', 'must be in _notebooks folder to init'
    os.chdir('..')
    __INIT_NOTEBOOK__ = True


# if '__INIT_NOTEBOOK__' not in globals():
#     with open('../_cells/_init_notebook.py') as handle:
#         exec(handle.read())
# from _util.util_v1 import * ; import _util.util_v1 as uutil
# from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
# from _util.twodee_v1 import * ; import _util.twodee_v1 as u2d
# import _util.flow_v1 as uflow
# import _util.sketchers_v2 as usketchers
# import _util.douga_v0 as udouga
# import _util.video_v1 as uvid
# import _util.distance_transform_v1 as udt
# device = torch.device('cuda')




