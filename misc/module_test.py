# if __name__ == "__main__":
#     from backbone.resnet import ResNet
#     from model.prompt_encoder import PromptEncoder
#     backbone = ResNet(18, pretrained=False,return_idx=[1,2,3])
#     model = PromptEncoder(backbone=backbone,num_classes=81,num_layers=3).cuda()
#     import torch
#     from thop import profile, clever_format
#     x = torch.randn(800, 3, 224, 224).cuda()
#     flops, params = profile(model, inputs=(x,))
#     flops, params = clever_format([flops, params], "%.3f")
#     print(flops, params)
if __name__ == '__main__':
    from backbone.resnet import ResNet
    from model.encoder import HybridEncoder
    from model.decoder import TideTransformer
    from model.tide import TIDE
    import torch
    import time
    from model.utils import NestedTensor
    model = TIDE(backbone=ResNet(18, pretrained=False,return_idx=[1,2,3]),
                    encoder=HybridEncoder(in_channels=[128, 256, 512]),
                    decoder=TideTransformer(num_classes=80,
                                            feat_channels=[256, 256, 256],
                                            ),
                    multi_scale=None)
    model.eval()
    x = torch.randn(4, 3, 224, 224)
    y = torch.randn(4,25, 384)
    x = NestedTensor(x,torch.zeros(4,224,224))
    y = NestedTensor(y,torch.ones((4, 25), dtype=torch.bool))
    inf_time = time.time()
    res = model(x,y)
    print(time.time()-inf_time)
    from thop import profile, clever_format
    flops, params = profile(model, inputs=(x,y))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params)
    