



from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
import _util.training_v1 as utraining


def infer_module_run(run_name):
    im = utraining.infer_module_script(run_name)
    exec(f'import {im.run.run_module} as _ModuleRun_{run_name}')
    return eval(f'_ModuleRun_{run_name}')

class Checkpoint:
    def __init__(self, run_name, ckpt=None):
        self.im_ckpt = utraining.infer_module_checkpoint(run_name, ckpt)
        self.module_run = infer_module_run(run_name)
        self.module_dataset = self.module_run.module_dataset
        self.module_model = self.module_run.module_model
        self.args = self.module_run.args
        return
    def sd(self):
        return Dict(torch.load(self.im_ckpt.ckpt_fn))
    def model(self):
        return self.module_model.Model.load_from_checkpoint(self.im_ckpt.ckpt_fn).eval()
    def dataset(self, split, force=None, *args, **kwargs):
        new_args = copy.deepcopy(self.args)
        if force is not None: new_args.update(force)
        return self.module_dataset.Dataset(args=new_args, *args, split=split, **kwargs)
    def datamodule(self, force=None, *args, **kwargs):
        new_args = copy.deepcopy(self.args)
        if force is not None: new_args.update(force)
        return self.module_dataset.Datamodule(args=new_args, *args, **kwargs)

