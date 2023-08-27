import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self,num_class:int) -> None:
        super(VGG16, self).__init__()
        
        self.num_class = num_class

        self.layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.layer_2= nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
           
        )
        
        self.layer_3= nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
           
        )
        
        
        self.layer_4= nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
           
        )
        
        self.layer_5= nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
           
        )
        
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=512*7*7,out_features=4096),
            nn.Linear(in_features=4096,out_features=4096),
            nn.Linear(in_features=4096,out_features=self.num_class),
        )
        
        
        self.softmax = nn.MaxPool2d(kernel_size=2,stride=2)
    def forward(self, x):
        # layer 1 + softmax 
        x = self.layer_1(x)
        x = self.softmax(x)
        
        # layer 2 + softmax 
        x = self.layer_2(x)
        x = self.softmax(x)
        
        # layer 3 + softmax 
        x = self.layer_3(x)
        x = self.softmax(x)
        
        # layer 4 + softmax 
        x = self.layer_4(x)
        x = self.softmax(x)
        
        # layer 5 + softmax 
        x = self.layer_5(x)
        x = self.softmax(x)
        
        # reshaping before passing it to classification layer
        x = x.view(x.size()[0], -1)
        
        # fully connected layer
        x = self.fc_layers(x)
        

        return x


