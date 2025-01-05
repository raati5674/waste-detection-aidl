import torch
from torch import nn
import torch


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=96, patch_size=16, num_hiddens=512):
        super().__init__()
        def _make_tuple(x):
            if not isinstance(x, (list, tuple)):
                return (x, x)
            return x
        
        img_size, patch_size = _make_tuple(img_size), _make_tuple(patch_size)
        self.num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=patch_size,stride=patch_size)

    def forward(self, X):
        # Output shape: (batch size, no. of patches, no. of channels)
        return self.conv(X).flatten(2).transpose(1, 2)


def test():
    img_size, patch_size, num_hiddens, batch_size = 96, 16, 512, 4
    patch_emb = PatchEmbedding(img_size, patch_size, num_hiddens)
    X = torch.zeros(batch_size, 3, img_size, img_size)
    patches=patch_emb(X)
    print("patches shape: ",patches.shape,"\n")
    print('hi')


#test()