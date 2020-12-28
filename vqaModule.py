import torch.nn as nn


class VqaModule(nn.Module):
    def __init__(self):
        super(VqaModule, self).__init__()
        
        # image stuff
        self.image_conv_layers = nn.Sequential(
            self.create_conv_layer(2, 3, 64),
            self.create_conv_layer(2, 64, 128),
            self.create_conv_layer(3, 128, 256),
            self.create_conv_layer(3, 256, 512),
            self.create_conv_layer(3, 512, 512) 
        )

        self.image_fc_layers = nn.Sequential(
            self.create_fc_layer(7 * 7 * 512, 4096),
            self.create_fc_layer(4096, 4096),
            self.create_fc_layer(4096, 1024),   
        )
       
        # question stuff
   
    def forward(self, im):
        # b = self.image_layer1(im)
        a = self.image_conv_layers(im)
        a = a.flatten()
        a = self.image_fc_layers(a)
        print (a.size())
        return a

    def create_conv_layer(self, number_of_layers, in_size, out_size):
        modules = []
        modules.append(nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1))

        for i in range(number_of_layers - 1):
            modules.append(nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1))

        modules.append(nn.ReLU())
        modules.append(nn.MaxPool2d(kernel_size=2, stride=2))        
        return nn.Sequential(*modules)

    def create_fc_layer(self, input_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU()
        )