


from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch

import _util.training_v1 as utraining

from _train.danbooru_tagger.models.kate import Model as DanbooruTagger
class ResnetFeatureExtractor(nn.Module):
    def __init__(
            self,
            path='./_train/danbooru_tagger/runs/waning_kate_vulcan0001/checkpoints/'
                'epoch=0022-val_f2=0.4461-val_loss=0.0766.ckpt',
            size_in=None,
            pca=None,  # ('./_data/lustrous/preprocessed/minna_resnet_feats_ortho_pca.pkl', ncomp)
    ):
        super().__init__()
        self.path = iq = path
        self.size_in = si = size_in
        # if iq[0]=='torchvision':
        if self.path in [None, 'uninitialized']:
            # use pytorch pretrained resnet50
            self.base_hparams = None
            resnet = tv.models.resnet50(pretrained={
                None: True,
                'uninitialized': False,
            }[self.path])

            self.resize = tv.transforms.Resize(256)
            self.resnet_preprocess = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu      #   64ch, 128p (assuming 256p input)
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1  #  256ch,  64p
            self.layer2 = resnet.layer2  #  512ch,  32p
            self.layer3 = resnet.layer3  # 1024ch,  16p
            self.layer4 = resnet.layer4  # 2048ch,   8p
            self.avgpool = resnet.avgpool  # 2048ch
            self.fc = resnet.fc            # 1062ch
        else:
            # use pretrained stoppa, danbooru-specific
            base = DanbooruTagger.load_from_checkpoint(self.path).eval()
            # base = userving.infer_model_load(*iq).eval()
            self.base_hparams = base.hparams
            self.rules = base.rules
            
            self.resize = tv.transforms.Resize(
                base.hparams.largs.danbooru_sfw.size
                if si is None else si
            )
            self.resnet_preprocess = base.resnet_preprocess
            self.conv1 = base.resnet.conv1
            self.bn1 = base.resnet.bn1
            self.relu = base.resnet.relu      #   64ch, 128p (assuming 256p input)
            self.maxpool = base.resnet.maxpool
            self.layer1 = base.resnet.layer1  #  256ch,  64p
            self.layer2 = base.resnet.layer2  #  512ch,  32p
            self.layer3 = base.resnet.layer3  # 1024ch,  16p
            self.layer4 = base.resnet.layer4  # 2048ch,   8p
            self.avgpool = base.resnet.avgpool
            self.fc = base.resnet.fc
        self.channels = [
            64,
            256,
            512,
            1024,
            2048,
        ]
        if pca!=None:
            self.pca_fn = pca[0]
            self.pca = uutil.pload(pca[0])
            self.pca_ncomp = pca[1]
            self.pca_weights = nn.Parameter(torch.tensor(self.pca.components_[:self.pca_ncomp])[None])
            self.pca_mean = nn.Parameter(torch.tensor(self.pca.mean_)[None])
        else:
            self.pca = None
        return
    def get_unfrozen(self, freeze):
        # get tunable params from freeze-index and onwards
        assert 0<=freeze<=5
        ret = []
        if freeze<=0:
            ret.append(self.conv1)
            ret.append(self.bn1)
        if freeze<=1:
            ret.append(self.layer1)
        if freeze<=2:
            ret.append(self.layer2)
        if freeze<=3:
            ret.append(self.layer3)
        if freeze<=4:
            ret.append(self.layer4)
            ret.append(self.pca_weights)
            ret.append(self.pca_mean)
        if freeze<=5:
            ret.append(self.fc)
        return ret
    def forward(self, x, return_more=False):
        x = x['image']
        ans = Dict()
        x = x[:,:3]
        x = self.resize(x)
        x = self.resnet_preprocess(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        ans['conv1'] = x  # conv1
        x = self.maxpool(x)
        x = self.layer1(x)
        ans['layer1'] = x  # layer1
        x = self.layer2(x)
        ans['layer2'] = x  # layer2
        x = self.layer3(x)
        ans['layer3'] = x  # layer3
        x = self.layer4(x)
        ans['layer4'] = x  # layer4
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        ans['avgpool'] = x  # avgpool
        x = self.fc(x)
        ans['fc'] = x  # fc
        if self.pca!=None:
            ans['pca'] = (
                self.pca_weights @ (ans['avgpool'] - self.pca_mean)[...,None]
            ).squeeze(-1)
        ans = Dict({
            'features': ans,
            'raw': ans['fc'],
        })
        if return_more:
            ans['prob'] = torch.sigmoid(ans['raw'])
            ans['pred'] = ans['raw']>0
            ans['prob_dict'] = [
                {
                    r['name']: p
                    for r,p in zip(self.rules, q)
                }
                for q in ans['prob'].detach().cpu().numpy()
            ]
            ans['locals'] = locals()
        return ans








