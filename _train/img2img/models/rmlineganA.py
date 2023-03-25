


from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.twodee_v1 import * ; import _util.twodee_v1 as u2d
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch

import _util.training_v1 as utraining
import _train.img2img.datasets.rmlineE as default_mdata


class Model(pl.LightningModule):
    default_args=Dict(
        **default_mdata.Datamodule.default_args,
        model=Dict(
            task=utraining.infer_module_model(uutil.fnstrip(__file__)).task_name,
            module=utraining.infer_module_model(uutil.fnstrip(__file__)).model_name,

            # images
            patch_size=9,

            # generator
            gen_depth=6,
            gen_width=32,
            gen_batchnorm=True,
            gen_use_hull=True,
            gen_mask_input=True,

            # discriminator
            dis_depth=4,
            dis_width=16,
            dis_batchnorm=True,
            dis_use_hull=True,
            
            # learning
            lerp_output=True,
            label_smoothing=0.8,
            lr_gen=0.001,
            lr_dis=0.001,
            lambda_l1=1.0,
            lambda_adv=1.0,

            # boilerplate
            precision=16,
            gradient_clip_val=None,
            checkpoint=Dict(
                every_n_epochs=10,
                filename='{epoch:04d}',
                save_top_k=-1,
                # monitor='val_loss',
                # mode='min',
                # filename='{epoch:04d}-{val_loss:0.8f}',
                # save_top_k=4,
            ),
        )
    )
    def __init__(self, args=None):
        super().__init__()
        self.args_user = copy.deepcopy(args or Dict())
        self.save_hyperparameters(Model.update_args(self.args_user))
        margs = self.hparams.model
        # assert 1 + 2*margs.gen_depth <= self.hparams.prep.size
        assert margs.patch_size + 2*margs.gen_depth <= self.hparams.prep.size

        # generator
        gen = []
        chin = 3
        if margs.gen_use_hull:
            chin += 1
        w = margs.gen_width
        for i in range(margs.gen_depth):
            gen.append(nn.Conv2d(
                chin if i==0 else w,
                w if i!=margs.gen_depth-1 else 3,
                3, stride=1, padding=0,
            ))
            if i!=margs.gen_depth-1:
                gen.append(nn.LeakyReLU())
                if margs.gen_batchnorm:
                    gen.append(nn.BatchNorm2d(w))
        gen.append(nn.Tanh())
        self.generator = nn.Sequential(*gen)

        # discriminator
        dis = []
        chin = 3
        if margs.dis_use_hull:
            chin += 1
        w = margs.dis_width
        for i in range(margs.dis_depth):
            dis.append(nn.Conv2d(
                chin if i==0 else w,
                w,
                3, stride=1, padding=0,
            ))
            if i!=margs.dis_depth-1:
                dis.append(nn.LeakyReLU())
                if margs.dis_batchnorm:
                    dis.append(nn.BatchNorm2d(w))
        self.discriminator = nn.Sequential(*dis)
        return
    @staticmethod
    def update_args(args):
        ans = copy.deepcopy(__class__.default_args)
        ans.update(copy.deepcopy(args or Dict()))
        return ans

    def forward(
                self, x,
                pad=True,
                return_more=False,
            ):
        margs = self.hparams.model

        # preprocess if needed
        img = x['image']
        mask = x['line_mask']
        fhull = x['face_hull']
        if margs.gen_mask_input:
            img = img * (1-mask)
        if margs.gen_use_hull:
            stackin = torch.cat([img, fhull], dim=1)
        else:
            stackin = img
        if pad:
            stackin = nn.functional.pad(
                stackin,
                (margs.gen_depth,)*4,
                mode='replicate',
            )

        # forward
        out = self.generator(stackin)
        ret = Dict({
            'image': out,
            'line_mask': mask,
            'face_hull': fhull,
        })
        if return_more:
            ret.update({
                'locals': locals(),
            })
        return ret

    def forward_discriminator(self, x, return_more=False):
        margs = self.hparams.model

        # preprocess if needed
        img = x['image']
        mask = x['line_mask']
        fhull = x['face_hull']
        if margs.dis_use_hull:
            stackin = torch.cat([img, fhull], dim=1)
        else:
            stackin = img
        stackin = nn.functional.pad(
            stackin,
            ((margs.patch_size-img.shape[-1])//2,)*4,
            mode='replicate',
        )

        # forward
        logits = self.discriminator(stackin).mean((1,2,3))
        ret = Dict({
            'logits': logits,
            'probability': logits.sigmoid(),
        })
        if return_more:
            ret.update({
                'locals': locals(),
            })
        return ret

    def loss(self, pred, gt, det=None, return_more=False):
        margs = self.hparams.model
        det = (not self.training) if det is None else (det)

        # lerp if needed
        if margs.lerp_output:
            pred['image'] = torch.lerp(
                gt['image'].to(pred['image'].dtype),
                pred['image'],
                gt['line_mask'].to(pred['image'].dtype),
            )

        # reconstruction loss
        loss_l1 = ( (pred['image'] - gt['image']) ).abs().mean((1,2,3))

        # adversarial loss
        outd = self.forward_discriminator(pred, return_more=return_more)
        sm = margs.label_smoothing
        loss_adv = nn.functional.binary_cross_entropy_with_logits(
            outd['logits'], gt['real_label']*sm+sm/2, reduction='none',
        )

        # combine
        lambda_l1 = margs.lambda_l1
        lambda_adv = margs.lambda_adv
        ret = Dict({
            'loss': lambda_l1*loss_l1 + lambda_adv*loss_adv,
            'loss_l1': loss_l1,
            'loss_adv': loss_adv,
        })
        if return_more:
            ret.update({
                'locals': locals(),
            })
        return ret
    def training_step(self, batch, batch_idx, optimizer_idx):
        # generator
        if optimizer_idx==0:
            # fakes only, flipped labels
            batch = {k: v[:,0] for k,v in batch.items()
                if isinstance(v, torch.Tensor)}
            batch['real_label'] = torch.ones_like(batch['real_label'])
            pred = self.forward(batch, return_more=False)
            loss = self.loss(pred, batch, return_more=False)
            loss_reduced = {k: v.mean() for k,v in loss.items()}
            for k,v in loss_reduced.items():
                self.log(f'train_gen_{k}', v.detach())
            return loss_reduced['loss']

        # discriminator
        elif optimizer_idx==1:
            # both fake and real, true labels
            batch = {k: v.view(-1,*v.shape[2:]) for k,v in batch.items()
                if isinstance(v, torch.Tensor)}
            pred = self.forward(batch, return_more=False)
            loss = self.loss(pred, batch, return_more=False)
            loss_reduced = {k: v.mean() for k,v in loss.items()}
            for k,v in loss_reduced.items():
                self.log(f'train_dis_{k}', v.detach())
            return loss_reduced['loss']
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # pred = self.forward(batch, return_more=False)
        # loss = self.loss(pred, batch, return_more=False)
        # loss_counted = {k: (v.detach().sum(), len(v)) for k,v in loss.items()}
        # return loss_counted
        bns = batch['bn']
        batch = {k: v[:,0] for k,v in batch.items()
            if isinstance(v, torch.Tensor)}
        pred = self.forward(batch, return_more=False)
        dt = pred['image'].dtype
        return {
            'bns': bns,
            'images': torch.lerp(
                batch['image'].to(dt),
                pred['image'],
                batch['line_mask'].to(dt),
            ),
        }
    # def validation_epoch_end(self, outputs):
    #     add = defaultdict(float)
    #     cnt = defaultdict(int)
    #     for out in outputs:
    #         for k,(v,c) in out.items():
    #             add[k] += v
    #             cnt[k] += c
    #     for k in add:
    #         self.log(f'val_{k}', add[k]/cnt[k])
    #     return
    def validation_epoch_end(self, outputs):
        for out in outputs:
            for bn,img in zip(out['bns'], out['images']):
                bn = f'{int(bn):04d}'
                self.logger.experiment.add_image(
                    f'image_{bn}',
                    img,
                    self.current_epoch,
                )
        # data = uutil.pload(f'./_data/lustrous/preprocessed/patches/rmlineEIA_test.pkl')
        # m = next(self.parameters()).data
        # with torch.no_grad():
        #     for (fn,_),img,hull,mask in zip(*[
        #         data[k] for k in ['bns', 'images', 'face_hulls', 'line_masks',
        #     ]]):
        #         rs,dtype,franch,idx,view = fnstrip(fn).split('--')
        #         img = torch.tensor(img).to(m.device).to(m.dtype)[None]
        #         hull = torch.tensor(hull).to(m.device).to(m.dtype)[None]
        #         mask = torch.tensor(mask).to(m.device).to(m.dtype)[None]
        #         out = self.forward({
        #             'image': img, 'face_hull': hull, 'line_mask': mask,
        #         })
        #         ans = torch.lerp(img, out['image'], mask)
        #         # print(ans[0].shape)
        #         self.logger.experiment.add_image(
        #             f'{franch}.{idx}',
        #             ans[0],
        #             self.current_epoch,
        #         )
        #         # I(ans).save(mkfile(''))
        return

    def configure_optimizers(self):
        margs = self.hparams.model
        optg = torch.optim.Adam(self.generator.parameters(), lr=margs.lr_gen)
        optd = torch.optim.Adam(self.discriminator.parameters(), lr=margs.lr_dis)
        return [optg, optd], []
        # tunable = [
        #     self.unet,
        # ]
        # opt = torch.optim.Adam(
        #     [
        #         param
        #         for net in tunable
        #         for param in net.parameters()
        #     ],
        #     lr=margs.lr,
        # )
        # return opt
        # sched = {
        #     'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
        #         opt,
        #         mode='min',
        #         factor=0.5,
        #         patience=margs.lr_plateau_patience,
        #         verbose=True,
        #     ),
        #     'monitor': 'val_loss',
        #     'interval': 'epoch',
        #     'frequency': 1,
        #     'strict': True,
        # }
        # return [opt,], [sched,]








