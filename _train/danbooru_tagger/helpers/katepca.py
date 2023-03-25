


from _util.util_v1 import * ; import _util.util_v1 as uutil
from _util.pytorch_v1 import * ; import _util.pytorch_v1 as utorch

# load chonker
from _train.danbooru_tagger.helpers.katebackbone import ResnetFeatureExtractor
class ResnetFeatureExtractorPCA(nn.Module):
    def __init__(self, fn_features, dim_out):
        super().__init__()
        self.fn_features = fn_features
        self.dim_out = dim_out
        self.rfe = ResnetFeatureExtractor(
            pca=(self.fn_features, self.dim_out),
        ).eval()
    def forward(self, img):
        pw = self.rfe.pca_weights[:,None,None]
        pb = self.rfe.pca_mean[...,None,None]
        img = img.bg('k').t()[None].to(pw.device)
        feats = self.rfe({
            'image': torch.cat([
                img,
                img.flip(dims=(-1,)),
            ], dim=0),
        }).features.layer4
        feats = (
            pw @ (feats - pb).permute(0,2,3,1)[...,None]
        ).squeeze(-1).permute(0,3,1,2)
        return feats





