import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = in_channels // 2

        self.g = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.theta = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, 1)
        self.W = nn.Conv2d(self.inter_channels, in_channels, 1)

        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        batch_size, C, H, W = x.shape
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # (B, C//2, H*W)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)  # (B, C//2, H*W)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)  # (B, H*W, C//2)

        f = torch.matmul(theta_x, phi_x)  # (B, C//2, C//2)
        f_div_C = f / (H * W)

        y = torch.matmul(f_div_C, g_x)  # (B, C//2, H*W)
        y = y.view(batch_size, self.inter_channels, H, W)
        W_y = self.W(y)
        z = W_y + x
        return self.bn(z)


class NbrilNet(nn.Module):
    def __init__(self):
        super(NbrilNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.se1 = SEBlock(8)  # Add SE block after first convolution

        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.non_local = NonLocalBlock(16)  # Add non-local block after the second convolution
        self.se2 = SEBlock(16)  # Add SE block after second convolution

        self.fc1 = nn.Linear(16 * 8 * 8, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 100)

    def forward(self, x):
        x = nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.se1(x)  # SE block after first convolution
        x = nn.functional.max_pool2d(x, 2)  # Max pooling

        x = nn.functional.relu(self.bn2(self.conv2(x)))
        x = self.non_local(x)  # Non-local block
        x = self.se2(x)  # SE block after second convolution
        x = nn.functional.max_pool2d(x, 2)

        x = x.reshape(x.size(0), -1)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#SE block + non local neural net