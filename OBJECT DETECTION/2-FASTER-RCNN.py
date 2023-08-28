# -----------------IMPORTS-----------------#
import torch
import torchvision
import torch.nn as nn

# -----------------MODEL BACKBONE-----------------#
resnet50_model = torchvision.models.resnet50(pretrained=True)
## WE ARE TAKING FIRST FOUR BLOCKS OF RESNET50 MODEL
recurrent_layer = list(resnet50_model.children())[:8]
backbone_model = nn.Sequential(*recurrent_layer)

## UNFREEZE ALL THE PARAMETERS
for param in backbone_model.named_parameters():
    param[1].requires_grad = True
# -----------------IMPORTS-----------------#
# -----------------IMPORTS-----------------#


if __name__ == "__main__":
    rand_img = torch.randint(
        high=255, low=0, size=(1, 3, 256, 256), dtype=torch.float32
    )
    print(rand_img.shape)
    output = backbone_model(rand_img)
    print(output.shape)
