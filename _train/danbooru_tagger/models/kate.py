

from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch
from _util.twodee_v1 import * ; import _util.twodee_v1 as u2d


class Model(pl.LightningModule):
    def __init__(self, bargs, pargs, largs, margs):
        super().__init__()
        self.hparams.bargs = bargs
        self.hparams.pargs = pargs
        self.hparams.largs = largs
        self.hparams.margs = margs
        self.save_hyperparameters()

        # read rulebook
        # see ./hack/snips/danbooru_intently_combatively.py for preprocessing
        self.filter_subset = largs.danbooru_sfw.filter_subset
        self.filter_rules = largs.danbooru_sfw.filter_rules
        self.fn_rules = f'{bargs.dn}/_data/danbooru/_filters/{self.filter_subset}_{self.filter_rules}_rules.json'
        self.rules = jread(self.fn_rules)

        # setup resnet
        self.resnet = tv.models.resnet50(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, len(self.rules))
        self.resnet_preprocess = TT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # loss
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        return
    def loss(self, gt, pred, weights=None, reduce=True, return_more=False):
        # expects gt is one-hot float
        # expects pred is pre-softmax values
        bce = self.bce(pred, gt)
        wbce = (1 if weights is None else weights) * bce
        ans = {'loss': wbce.mean() if reduce else wbce}
        if return_more:
            with torch.no_grad():
                bing,binp = gt>0.5, pred>0
                ans['tp'] = ( bing& binp).float()
                ans['fn'] = ( bing&~binp).float()
                ans['fp'] = (~bing& binp).float()
                ans['tn'] = (~bing&~binp).float()
        return ans
    def forward(self, rgb, return_more=True):
        normed = self.resnet_preprocess(rgb)
        out_raw = self.resnet(normed)
        out = {'raw': out_raw}
        if return_more:
            out['prob'] = torch.sigmoid(out_raw)
            out['pred'] = out_raw>0
        return out
    
    def training_step(self, batch, batch_idx):
        # unpack
        rgb = batch['image']
        lab = batch['label']
        weights = batch['weights']
        
        # predict
        pred = self.forward(rgb, return_more=False)
        loss = self.loss(lab, pred['raw'], weights=weights, reduce=True, return_more=False)
        
        # log
        self.log('train_loss', loss['loss'])
        return {
            'loss': loss['loss'],
        }
    def validation_step(self, batch, batch_idx):
        # unpack
        rgb = batch['image']
        lab = batch['label']
        weights = batch['weights']
        
        # predict
        pred = self.forward(rgb, return_more=False)
        loss = self.loss(lab, pred['raw'], weights=weights, reduce=True, return_more=True)
        
        # log
        self.log('val_loss', loss['loss'])
        return {
            'val_loss': loss['loss'],
            '_tp': loss['tp'].sum(0),
            '_tn': loss['tn'].sum(0),
            '_fp': loss['fp'].sum(0),
            '_fn': loss['fn'].sum(0),
            '_cnt': len(rgb),
        }
    def validation_epoch_end(self, outputs):
        labels = [l for l in outputs[0].keys() if l.startswith('val_')]
        ans = {l: 0 for l in labels}
        for output in outputs:
            for label in labels:
                ans[label] += output[label]
        for label in labels:
            ans[label] /= len(outputs)
        for k,v in ans.items():
            self.log(k, v)

        cnt=tp=tn=fp=fn = 0
        for o in outputs:
            cnt += o['_cnt']
            tp += o['_tp']
            tn += o['_tn']
            fp += o['_fp']
            fn += o['_fn']
        tp /= cnt
        tn /= cnt
        fp /= cnt
        fn /= cnt
        # cnt = sum([o['_cnt'] for o in outputs])
        # tp = torch.stack([o['_tp'] for o in outputs]).sum(0) / cnt
        # tn = torch.stack([o['_tn'] for o in outputs]).sum(0) / cnt
        # fp = torch.stack([o['_fp'] for o in outputs]).sum(0) / cnt
        # fn = torch.stack([o['_fn'] for o in outputs]).sum(0) / cnt
        acc = tp + tn
        rec = tp / (tp+fn) ; rec[rec!=rec] = 1.0
        pre = tp / (tp+fp) ; pre[pre!=pre] = 0.0
        f1 = 2*tp/(2*tp+fn+fp) ; f1[f1!=f1] = 1.0
        f2 = 5*tp/(5*tp+4*fn+fp) ; f2[f2!=f2] = 1.0
        f5 = 1.5*tp/(1.5*tp+0.25*fn+fp) ; f5[f5!=f5] = 1.0
        self.log('val_acc', acc.mean())
        self.log('val_rec', rec.mean())
        self.log('val_pre', pre.mean())
        self.log('val_f1', f1.mean())
        self.log('val_f2', f2.mean())
        self.log('val_f5', f5.mean())
        return
    
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.margs.lr,
        )
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=100,
            eta_min=0,
        )
        return [opt,], [sched,]
