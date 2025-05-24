import torch
import torch.nn as nn
import timm
from timm.layers import trunc_normal_


# TODO: change hardcoded values
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
        # B, C, H, W = x.shape
        # assert (
        #     H == self.img_size[0] and W == self.img_size[1]
        # ), f"input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})"
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Model(nn.Module):
    def __init__(self, num_classes=101, lf=10, num_bottlenecks=4):
        super().__init__()

        # set num_classes=0 to remove classification head
        self.vv = timm.create_model(
            "vit_base_patch16_224.augreg_in21k", pretrained=True, num_classes=0
        )

        self.va = timm.create_model(
            "vit_base_patch16_224.augreg_in21k", pretrained=True, num_classes=0
        )

        # apply ast weights to va
        ast_weights = torch.load(
            "pretrained_weights/audioset_16_16_0.4422.pth", weights_only=True
        )
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

        # interpolate position embeddings for video
        num_patches_video = self.vv.patch_embed.num_patches
        self.vv.pos_embed = self.interpolate_pos_encoding(
            self.vv.pos_embed, num_patches_video
        )

        # interpolate position embeddings for audio
        num_patches_audio = self.va.patch_embed.num_patches
        self.va.pos_embed = self.interpolate_pos_encoding(
            self.va.pos_embed, num_patches_audio
        )

        self.lf = lf
        self.num_features = self.vv.embed_dim

        # create new classification head for fused features
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.num_features),
            nn.Linear(self.num_features, num_classes),
        )

        # create bottleneck fusion tokens
        self.num_bottlenecks = num_bottlenecks
        self.zfsn = nn.Parameter(torch.zeros(1, num_bottlenecks, self.vv.embed_dim))
        trunc_normal_(self.zfsn, std=0.02)

    def interpolate_pos_encoding(self, pos_embed, num_patches):
        pos_embed = pos_embed.float()

        # handle CLS token separately
        cls_pos_embed = pos_embed[:, 0:1, :]
        pos_embed = pos_embed[:, 1:, :]

        # interpolate patch position embeddings
        pos_embed = pos_embed.permute(0, 2, 1)
        pos_embed = nn.functional.interpolate(
            pos_embed, size=num_patches, mode="linear", align_corners=False
        )
        pos_embed = pos_embed.permute(0, 2, 1)

        # recombine with CLS token
        pos_embed = torch.cat((cls_pos_embed, pos_embed), dim=1)
        return nn.Parameter(pos_embed)

    def forward_features(self, x, v, lf):
        B = x.shape[0]
        x = v.patch_embed(x)
        if len(x.shape) > 3:  # just in case
            x = x.flatten(2).transpose(1, 2)
        cls_token = v.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = v.pos_drop(x + v.pos_embed)

        for i, block in enumerate(v.blocks):
            if i < lf:
                x = block(x)
        return x

    def forward(self, video, audio):
        B, F, C, H, W = video.shape
        video = video.view(B * F, C, H, W)

        # process separately until fusion layer
        v_features = self.forward_features(video, self.vv, self.lf)
        v_features = v_features.view(B, F, -1, self.num_features)
        v_features = torch.mean(v_features, dim=1)
        a_features = self.forward_features(audio, self.va, self.lf)

        # expand fusion tokens for batch
        zfsn = self.zfsn.expand(B, -1, -1)

        # process remaining layers with bottleneck fusion
        for block in self.vv.blocks[self.lf :]:
            # eqn 8
            v_concat = torch.cat([v_features, zfsn], dim=1)
            a_concat = torch.cat([a_features, zfsn], dim=1)

            v_out = block(v_concat)
            a_out = block(a_concat)

            # split features and fusion tokens
            v_features = v_out[:, : v_features.shape[1]]
            a_features = a_out[:, : a_features.shape[1]]
            v_zfsn = v_out[:, v_features.shape[1] :]
            a_zfsn = a_out[:, a_features.shape[1] :]

            # eqn 9
            zfsn = (v_zfsn + a_zfsn) / 2

        # get CLS tokens
        v_cls = v_features[:, 0]
        a_cls = a_features[:, 0]

        v_logits = self.classifier(v_cls)
        a_logits = self.classifier(a_cls)

        output = (v_logits + a_logits) / 2

        return output
