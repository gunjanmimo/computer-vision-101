# %%
import torch.nn as nn
from typing import Any, List


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        stride: int = 1,
        padding: int = 1,
        down_sample: Any = None,
    ) -> None:
        super(ResidualBlock, self).__init__()

        self.down_sample = down_sample

        self.layer_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=3,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
        )
        self.layer_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=3,
                stride=stride,
                padding=padding,
            ),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        residual = x
        out = self.layer_1(x)
        out = self.layer_2(out)

        if self.down_sample:
            residual = self.down_sample(x)
        out += residual
        out = nn.ReLU(out)
        return out


class Resnet50(nn.Module):
    def __init__(self, layers: List[int], num_class: int) -> None:
        super(Resnet50, self).__init__()
        self.inplanes = 64

        self.conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=self.inplanes,
                kernel_size=7,
                padding=3,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # create blocks
        self.res_block_1 = self._make_res_block(
            out_channel=64, num_block=layers[0], stride=1
        )
        self.res_block_2 = self._make_res_block(
            out_channel=128, num_block=layers[1], stride=2
        )
        self.res_block_3 = self._make_res_block(
            out_channel=256, num_block=layers[2], stride=2
        )
        self.res_block_4 = self._make_res_block(
            out_channel=512, num_block=layers[3], stride=2
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_class)

    def _make_res_block(self, out_channel: int, num_block: int, stride: int = 1):
        down_sample = None
        if stride != 1 and self.inplanes != out_channel:
            down_sample = nn.Sequential(
                nn.Conv2d(
                    in_channels=self.inplanes,
                    out_channels=out_channel,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(num_features=out_channel),
            )
        layers = []
        layers.append(
            ResidualBlock(
                in_channel=self.inplanes,
                out_channel=out_channel,
                stride=stride,
                down_sample=down_sample,
            )
        )
        self.inplanes = out_channel
        for i in range(1, num_block):
            layers.append(
                ResidualBlock(in_channel=self.inplanes, out_channel=out_channel)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.maxpool(x)

        x = self.res_block_1(x)
        x = self.res_block_2(x)
        x = self.res_block_3(x)
        x = self.res_block_4(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    import torch

    rand_image = torch.randint(
        low=0, high=255, size=(1, 3, 224, 224), dtype=torch.float32
    )
    print(rand_image.shape)
    print("-" * 100)
    model = Resnet50(layers=[3, 4, 6, 3], num_class=10)
    output = model(rand_image)
    print(output.shape)
# %%
