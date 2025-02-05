import torch
import torch.nn as nn
import timm


class Model(nn.Module):
    def __init__(self, num_classes=101, lf=8):
        super().__init__()

        # set num_classes=0 to remove classification head
        self.vv = timm.create_model(
            "vit_base_patch16_224.augreg_in21k", pretrained=True, num_classes=0
        )

        self.va = timm.create_model(
            "vit_base_patch16_224.augreg_in21k", pretrained=True, num_classes=0
        )
        
        # apply ast weights to va
        ast_weights = torch.load('pretrained_weights/audioset_10_10_0.4593.pth')
        temp = self.va.state_dict()
        pretrained_dict = {}
        for k, v in ast_weights.items():
            if k.startswith('module.'):
                k=k[7:]
            if k in temp and temp[k].shape == v.shape:
                pretrained_dict[k] = v
        temp.update(pretrained_dict)
        self.va.load_state_dict(temp)

        # for single channel audio input
        self.va.patch_embed.proj = nn.Conv2d(
            1,
            self.va.patch_embed.proj.out_channels,
            kernel_size=self.va.patch_embed.proj.kernel_size,
            stride=self.va.patch_embed.proj.stride,
            padding=self.va.patch_embed.proj.padding,
        )

        self.lf = lf
        self.num_features = self.vv.embed_dim

        # create new classification head for fused features
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features, self.num_features),
            nn.LayerNorm(self.num_features),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.num_features, num_classes),
        )

    def forward_features(self, x, v, lf):
        B=x.shape[0]
        x = v.patch_embed(x)
        if len(x.shape)>3:
            x = x.flatten(2).transpose(1,2)
        cls_token = v.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = v.pos_drop(x + v.pos_embed)

        for i, block in enumerate(v.blocks):
            if i < lf:
                x = block(x)
        return x

    def forward(self, video, audio):
        B, F, c, h, w = video.shape
        video = video.view(B*F, c, h, w)
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
        a_cls = fused[:, self.vv.patch_embed.num_patches + 1]
        
        v_logits = self.classifier(v_cls)
        a_logits = self.classifier(a_cls)
        
        output = (v_logits+a_logits)/2

        return output
