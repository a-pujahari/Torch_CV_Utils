
import torch
import torch.nn as nn
import torch.nn.functional as F

class Ultimus(nn.Module):
    def __init__(self):
        super().__init__()

        self.keys = nn.Linear(48, 8)
        self.queries = nn.Linear(48,8)
        self.values = nn.Linear(48,8)

        self.scale = 8 ** -0.5
        self.attend = nn.Softmax(dim = -1)
        self.outputFC = nn.Linear(8,48)

    def forward(self, x):
        K = self.keys(x)
        Q = self.queries(x)
        V = self.values(x)

        dot_product = torch.matmul(Q.transpose(-1, -2), K) * self.scale
        attention = self.attend(dot_product)
        Z = torch.matmul(V, attention)

        out = self.outputFC(Z)
        return out


class net(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.convBlock = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=0, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=256, kernel_size=3, stride=1, padding=0, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(in_channels=256, out_channels=48, kernel_size=3, stride=1, padding=0, bias=False),
            nn.GELU(),
            nn.BatchNorm2d(48),
        )

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.ultimus1 = Ultimus()
        self.ultimus2 = Ultimus()
        self.ultimus3 = Ultimus()
        self.ultimus4 = Ultimus()

        self.FC = nn.Linear(48,10)

    def forward(self, x):
        out = self.convBlock(x)
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.ultimus1(out)
        out = self.ultimus2(out)
        out = self.ultimus3(out)
        out = self.ultimus4(out)

        out = self.FC(out)
        return out
