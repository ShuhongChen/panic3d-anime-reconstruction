


from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch



#################### infer values ####################

def infer_num_workers(num_workers):
    if isinstance(num_workers, int):
        ans = num_workers
    elif num_workers=='max':
        ans = len(os.sched_getaffinity(0))
    return max(1, ans)
def infer_batch_size(bs, num_gpus, max_bs_per_gpu):
    # inputs:
    #     bs: effective batch size
    #     num_gpus: num gpus available, pretend 1 if cpu
    #     max_bs_per_gpu: max samples per gpu at once, none if don't care
    # constraints:
    #     bs = num_gpus * accumulate_grad_batches * bs_realized
    #     bs_realized <= max_bs_per_gpu
    num_gpus = max(1, num_gpus)
    max_bs_per_gpu = bs if max_bs_per_gpu is None else max_bs_per_gpu

    bsr = bs//num_gpus
    if bsr!=bs/num_gpus:
        warnings.warn(f'batch_size={bs} not divisible by num_gpus={num_gpus}')
    agd = 1
    while bsr > max_bs_per_gpu:
        if bsr//2!=bsr/2:
            warnings.warn(f'batch_size accumulation not divisible')
        bsr = bsr//2
        agd = agd*2
    return {
        'bs_realized': bsr,
        'accumulate_grad_batches': agd,
    }


#################### infer modules ####################

def infer_module_task(query, bargs=Dict(dn='.')):
    # query: task name
    tdn = f'{bargs["dn"]}/_train'
    if os.path.isdir(f'{tdn}/{query}'):
        return Dict(
            task_name=query,
            task_dn=f'{tdn}/{query}',
            task_module=f'_train.{query}',
        )
def infer_module_dataset(query, bargs=Dict(dn='.')):
    # query: dataset name or file name
    tdn = f'{bargs["dn"]}/_train'
    if os.path.isfile(query):
        query = query.split('/')[-1].split('.')[0]
    q = query
    for task in os.listdir(tdn):
        if os.path.isfile(f'{tdn}/{task}/datasets/{q}.py'):
            return Dict(
                dataset_name=q,
                dataset_module=f'_train.{task}.datasets.{q}',
                dataset_fn=f'{tdn}/{task}/datasets/{q}.py',
                task_name=task,
            )
    return None
def infer_module_model(query, bargs=Dict(dn='.')):
    # query: model name or file name
    tdn = f'{bargs["dn"]}/_train'
    if os.path.isfile(query):
        query = query.split('/')[-1].split('.')[0]
    q = query
    for task in os.listdir(tdn):
        if os.path.isfile(f'{tdn}/{task}/models/{q}.py'):
            return Dict(
                model_name=q,
                model_module=f'_train.{task}.models.{q}',
                model_fn=f'{tdn}/{task}/models/{q}.py',
                task_name=task,
            )
    return None
def infer_module_trainer(query, bargs=Dict(dn='.')):
    # query: trainer name or file name
    tdn = f'{bargs["dn"]}/_train'
    if os.path.isfile(query):
        query = query.split('/')[-1].split('.')[0]
    q = query
    for task in os.listdir(tdn):
        if os.path.isfile(f'{tdn}/{task}/trainers/{q}.py'):
            return Dict(
                trainer_name=q,
                trainer_module=f'_train.{task}.trainers.{q}',
                trainer_fn=f'{tdn}/{task}/trainers/{q}.py',
                task_name=task,
                dataset_name=q.split('_')[0],
                model_name=q.split('_')[1],
            )
    return None
def infer_module_run(query, bargs=Dict(dn='.')):
    # query: name or fn of py/sh
    tdn = f'{bargs["dn"]}/_train'
    if os.path.isfile(query):
        query = query.split('/')[-1].split('.')[0]
    q = query
    spl = q.split('_')
    for task in os.listdir(tdn):
        dn = f'{tdn}/{task}/runs/{q}'
        if os.path.isdir(dn):
            cdn = f'{dn}/checkpoints'
            epos = []
            epos_min = None
            epos_max = None
            if os.path.isdir(cdn):
                cfns = os.listdir(cdn)
                epos = sorted([
                    (
                        f'{cdn}/{fn}',
                        int(fn.split('=')[1].split('-')[0].split('.')[0]),
                        float(fn.split('=')[-1][:-len('.ckpt')]),
                    )
                    for fn in cfns
                    if fn.endswith('.ckpt')
                    and fn.startswith('epoch=')
                ], key=lambda q: q[-1])
                if len(epos)>0:
                    epos_min = epos[0]
                    epos_max = epos[-1]
            return Dict(
                run_name=q,
                run_module=f'_train.{task}.runs.{q}.{q}',
                run_ver=spl[-1],
                run_dn=dn,
                run_fn_sh=f'{dn}/{q}.sh',
                run_fn_py=f'{dn}/{q}.py',
                run_fn_help=f'{dn}/{q}.txt',
                run_dn_checkpoints=cdn,
                run_fn_checkpoint=f'{cdn}/last.ckpt',
                run_epochs=sorted(epos),
                run_epoch_min=epos_min,
                run_epoch_max=epos_max,
                run_fn_key_neptune=f'{dn}/neptune.key',
                run_fn_key_wandb=f'{dn}/wandb.key',
                run_dn_logs=f'{dn}/logs',
                task_name=task,
                dataset_name=spl[0],
                model_name=spl[1],
                trainer_name='_'.join(spl[:2]),
            )
    return None
def infer_module_script(query, bargs=Dict(dn='.')):
    # query: __file__ of run py
    tdn = f'{bargs["dn"]}/_train'
    q = query.split('/')[-1].split('.')[0]
    spl = q.split('_')
    for task in os.listdir(tdn):
        dn = f'{tdn}/{task}/runs/{q}'
        if os.path.isdir(dn):
            return Dict(
                task=infer_module_task(task),
                dataset=infer_module_dataset(spl[0]),
                model=infer_module_model(spl[1]),
                trainer=infer_module_trainer('_'.join(spl[:2])),
                run=infer_module_run(q),
            )
    return None
def infer_module_checkpoint(run, ckpt, bargs=Dict(dn='.')):
    # run name + ckpt filename
    if ckpt is None: return None
    ir = infer_module_run(run, bargs=bargs)
    if isinstance(ckpt, str) and ckpt.isnumeric():
        ckpt = int(ckpt)
    if isinstance(ckpt, str):
        if ckpt in ['min', 'max']:
            ire = ir[f'run_epoch_{ckpt}']
            assert ire is not None, 'no parsable checkpoints'
            fn = ire[0]
        else:
            fns = [
                f'{ir.run_dn_checkpoints}/{fn}' for fn in os.listdir(ir.run_dn_checkpoints)
                if fn.startswith(ckpt)
            ]
            assert len(fns)==1, 'ambiguous checkpoints to load'
            fn = fns[0]
    elif isinstance(ckpt, int):
        if ckpt==-1: return infer_module_checkpoint(run, 'last.ckpt', bargs=bargs)
        p = re.compile(r'\d+')
        fn = {
            int(p.search(fn).group()): f'{ir.run_dn_checkpoints}/{fn}'
            for fn in os.listdir(ir.run_dn_checkpoints)
            if fn.endswith('.ckpt') and p.search(fn) is not None
        }[ckpt]
    ckpt = fn
    fstr = uutil.fnstrip(ckpt)
    epoch = int(fstr.split('=')[1].split('-')[0]) if fstr.startswith('epoch=') else -1
    return Dict(
        ckpt_fn=fn,
        ckpt_name=fstr,
        ckpt_epoch=epoch,
        task_name=ir.task_name,
        dataset_name=ir.dataset_name,
        model_name=ir.model_name,
        run_name=ir.run_name,
    )


#################### loggers ####################

def logger_wandb(args, inferred_run, offline_mode=False, write_key=True):
    im = inferred_run
    os.environ['WANDB_START_METHOD'] = 'thread'
    wandb.login(key=os.environ['WANDB_API_TOKEN'])
    mkdir(f'{im.run.run_dn_logs}/wandb')
    eid = read(im.run.run_fn_key_wandb) \
        if os.path.isfile(im.run.run_fn_key_wandb) \
        else None
    ans = pl.loggers.wandb.WandbLogger(
        project=args.base.project,
        name=im.run.run_name,
        save_dir=im.run.run_dn_logs,
        version=eid,
        offline=offline_mode,
        log_model=False,
        experiment=None,
        prefix='',
    )
    if write_key and eid is None:
        uutil.write(ans.experiment.id, im.run.run_fn_key_wandb)
    return ans
def logger_neptune(args, inferred_run, offline_mode=False, write_key=True):
    # bug: on first run, has memory leak around iteration 50
    im = inferred_run
    eid = read(im.run.run_fn_key_neptune) \
        if os.path.isfile(im.run.run_fn_key_neptune) \
        else None
    ans = pl.loggers.NeptuneLogger(
        api_key=os.environ['NEPTUNE_API_TOKEN'],
        project_name=f'{os.environ["NEPTUNE_USER"]}/{args.base.project}',
        offline_mode=offline_mode,
        experiment_name=im.run.run_name,
        experiment_id=eid,
        prefix='',
        close_after_fit=True,
    )
    if write_key and eid is None:
        uutil.write(ans.experiment.id, im.run.run_fn_key_neptune)
    return ans
def logger_tensorboard(args, inferred_run):
    im = inferred_run
    ans = pl.loggers.TensorBoardLogger(
        im.run.run_dn_logs,
        name='tensorboard',
        version=0,
        log_graph=False,
        default_hp_metric=True,
        prefix='',
    )
    return ans




