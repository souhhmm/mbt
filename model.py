import torch
import torch.nn as nn
import timm
from timm.layers import trunc_normal_

# TODO change hardcoded values
class PatchEmbed(nn.Module):
    def __init__(
        self, img_size=(128, 400), patch_size=(16, 16), in_chans=1, embed_dim=768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})"
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Model(nn.Module):
    def __init__(self, num_classes=101, lf=10):
        super().__init__()

        # set num_classes=0 to remove classification head
        self.vv = timm.create_model(
            "vit_base_patch16_224.augreg_in21k", pretrained=True, num_classes=0
        )

        self.va = timm.create_model(
            "vit_base_patch16_224.augreg_in21k", pretrained=True, num_classes=0
        )

        # apply ast weights to va
        ast_weights = torch.load("pretrained_weights/audioset_16_16_0.4422.pth")
        temp = self.va.state_dict()
        pretrained_dict = {}
        for k, v in ast_weights.items():
            if k.startswith("module."):
                k = k[7:]
            if k in temp and temp[k].shape == v.shape:
                pretrained_dict[k] = v
        temp.update(pretrained_dict)
        self.va.load_state_dict(temp)

        self.va.patch_embed = PatchEmbed(
            img_size=(128, 400),
            patch_size=(16, 16),
            in_chans=1,
            embed_dim=self.va.embed_dim,
        )

        num_patches = self.va.patch_embed.num_patches
        self.va.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, self.va.embed_dim)
        )

        trunc_normal_(self.va.pos_embed, std=.02)
        
        self.lf = lf
        self.num_features = self.vv.embed_dim

        # create new classification head for fused features
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Linear(self.num_features, num_classes),
        )

    def forward_features(self, x, v, lf):
        B = x.shape[0]
        x = v.patch_embed(x)
        if len(x.shape) > 3:
            x = x.flatten(2).transpose(1, 2)
        cls_token = v.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = v.pos_drop(x + v.pos_embed)

        for i, block in enumerate(v.blocks):
            if i < lf:
                x = block(x)
        return x

    def forward(self, video, audio):
        B, F, c, h, w = video.shape
        video = video.view(B * F, c, h, w)
        # process separately until fusion layer
        v_features = self.forward_features(video, self.vv, self.lf)
        v_features = v_features.view(B, F, -1, self.num_features)
        v_features = torch.mean(v_features, dim=1)
        a_features = self.forward_features(audio, self.va, self.lf)

        # concatenate features
        fused = torch.cat((v_features, a_features), dim=1)

        for i in range(self.lf, len(self.vv.blocks)):
            fused = self.vv.blocks[i](fused)

        v_cls = fused[:, 0]
        a_cls = fused[:, self.va.patch_embed.num_patches + 1]

        v_logits = self.classifier(v_cls)
        a_logits = self.classifier(a_cls)

        output = (v_logits + a_logits) / 2

        return output
