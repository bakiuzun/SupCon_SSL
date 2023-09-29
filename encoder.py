import torch
from torch import nn, Tensor
import torch.nn.functional as F
import ot

from torchvision.models import resnet18,resnet50


class WrappedResnet(torch.nn.Module):
    output_size = 128
    def __init__(self):
        super().__init__()
        N_CHANNELS = 64
        self.sentinel1_block = torch.nn.Sequential(
            torch.nn.Conv2d(2, N_CHANNELS, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(N_CHANNELS),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(3, 2),
        )
        self.sentinel2_block = torch.nn.Sequential(
            torch.nn.Conv2d(4, N_CHANNELS, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(N_CHANNELS),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(3, 2),
        )
        self.model = resnet18(num_classes=128)
        self.model.conv1 = torch.nn.Identity()

    def forward(self, x):
        if x.size(1) == 2:
            x = self.sentinel1_block(x)
        else:
            x = self.sentinel2_block(x)
        return self.model(x)



def batch_sinkhorn(
    a: Tensor, b: Tensor, C: Tensor, reg: float, max_iters: int = 10,
) -> Tensor:
    """
    Solve a batch of Entropically regularized optimal transport problems
    using the Sinkhorn-Knopp algorithm.

    Parameters
    ==========
        a: Tensor - size (n1,)
        b: Tensor - size (n2,)
        C: Tensor - size (b,n1,n2)
        reg: float - entropic regularization (lambda)
        max_iters: int - the number of iterations
    Returns
    =======
        plans: Tensor - size (b,n1,n2) optimal transport plans
    """
    K = (-C / reg).exp()

    u = torch.ones_like(a)
    for _ in range(max_iters):
        v = b / torch.einsum("...ij,...i", K, u)
        u = a / torch.einsum("...ij,...j", K, v)
    return u.unsqueeze(-1) * K * v.unsqueeze(-2)


@torch.no_grad()
def batch_transport(a: Tensor, b: Tensor, C: Tensor) -> Tensor:
    """
    Solve a batch of Entropically regularized optimal transport problems
    using the Sinkhorn-Knopp algorithm.

    Parameters
    ==========
        a: Tensor - size (n1,)
        b: Tensor - size (n2,)
        C: Tensor - size (b,n1,n2)
    Returns
    =======
        plans: Tensor - size (b,n1,n2) optimal transport plans
    """
    batch_size = C.size(0)
    gammas = torch.empty_like(C, device="cpu")

    for i in range(batch_size):
        gammas[i, :, :] = ot.emd(a, b, C[i, :, :].cpu())

    return gammas.to(C.device)


class L2Norm(nn.Module):
    def forward(self, x):
        return x / x.norm(p=2, dim=1, keepdim=True)


class SmallAlexNet(nn.Module):
    def __init__(self, in_channel=3, feat_dim=128):
        super().__init__()

        blocks = []

        # conv_block_1
        blocks.append(nn.Sequential(
            nn.Conv2d(in_channel, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))

        # conv_block_2
        blocks.append(nn.Sequential(
            nn.Conv2d(96, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))

        # conv_block_3
        blocks.append(nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        ))

        # conv_block_4
        blocks.append(nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        ))

        # conv_block_5
        blocks.append(nn.Sequential(
            nn.Conv2d(384, 192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        ))

        # fc6
        blocks.append(nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * 8 * 8, 4096, bias=False),  # 256 * 6 * 6 if 224 * 224
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        ))

        # fc7
        blocks.append(nn.Sequential(
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        ))

        # fc8
        blocks.append(nn.Sequential(
            nn.Linear(4096, feat_dim),
            L2Norm(),
        ))

        self.blocks = nn.ModuleList(blocks)
        self.init_weights_()

    def init_weights_(self):
        def init(m):
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0, 0.02)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                if getattr(m, 'weight', None) is not None:
                    nn.init.ones_(m.weight)
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init)

    def forward(self, x, *, layer_index=-1):
        if layer_index < 0:
            layer_index += len(self.blocks)
        for layer in self.blocks[:(layer_index + 1)]:

            x = layer(x)
        return x


class SharedNetWithPrototypes(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SmallAlexNet()
        self.encoder.blocks[0] = nn.Identity()

        self.sentinel1_block = nn.Sequential(
            nn.Conv2d(2, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.sentinel2_block = nn.Sequential(
            nn.Conv2d(4, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )

        N_PROTOTYPES = 300
        EMBED_DIM=96
        self.register_parameter(
            "prototypes",
            torch.nn.parameter.Parameter(0.01 * torch.randn((N_PROTOTYPES, EMBED_DIM))),
        )

    def normalize_prototypes(self):
        w = self.prototypes.data.clone()
        w = torch.nn.functional.normalize(w, dim=-1)
        self.prototypes.data.copy_(w)

    def forward(self, x):
        if x.size(1) == 2:
            x = self.sentinel1_block(x)
        else:
            x = self.sentinel2_block(x) # (b,c,h,w)

        b, c, h, w, = x.shape
        mu = torch.full((h*w,), 1/(h*w), device=x.device)
        p = self.prototypes.size(0)
        nu = torch.full((p,), 1/p, device=x.device)

        x = x.permute(0, 2, 3, 1).view(b, h * w, c)
        S = torch.nn.functional.normalize(x, dim=-1) @ self.prototypes.T

        gamma = batch_sinkhorn(mu, nu, 1-S, reg=0.1, max_iters=10)
        x = x.size(0) * gamma @ self.prototypes # Barycentric projections, (b,h*w,c)
        x = x.permute(0, 2, 1).view(b, c, h, w)

        return self.encoder(x)



class DualEncoder(nn.Module):
    output_size = 128

    # use one model for each modality
    def __init__(self):
        super().__init__()
        self.encoder_1 = SharedNet()
        self.encoder_2 = SharedNet()

    def forward(self, x):
        if x.size(1) == 2:
            x = self.encoder_1(x)
        else:
            x = self.encoder_2(x)
        return x


class SharedNet(nn.Module):
    output_size = 128

    def __init__(self,return_block_act: bool = False):
        super().__init__()
        self.encoder = SmallAlexNet()
        self.encoder.blocks[0] = nn.Identity()
        self.return_block_act = return_block_act

        self.sentinel1_block = nn.Sequential(
            nn.Conv2d(2, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
        self.sentinel2_block = nn.Sequential(
            nn.Conv2d(4, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
        )
       

    def forward(self, x):
        assert x.size(1) in (2,4)
        if x.size(1) == 2:
            out = self.sentinel1_block(x)
        else:
            out = self.sentinel2_block(x)

        act = self.encoder(out)
        if self.return_block_act:
            return act, out

        return act




