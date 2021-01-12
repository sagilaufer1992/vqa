import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VqaModule(nn.Module):
    def __init__(self):
        super(VqaModule, self).__init__()
        
        # image stuff
        self.image_conv_layers = nn.Sequential(
            self.create_conv_layer(2, 3, 64),
            # self.create_conv_layer(1, 64, 128),
            self.create_conv_layer(2, 64, 256),
            # self.create_conv_layer(1, 256, 512),
            self.create_conv_layer(1, 256, 512) 
        )

        self.image_fc_layers = nn.Sequential(
            self.create_fc_layer(8 * 8 * 512, 4096),
            self.create_fc_layer(4096, 1024),
            # self.create_fc_layer(2048, 512),   
        )
       
        # question stuff

        self.embedding = nn.Embedding(8192, 256)
        self.lstm = nn.LSTM(256, 1024, 2)


        self.mlp1 = torch.nn.Linear(1024, 32)
        # self.mlp1 = torch.nn.Linear(512, 256)
        self.mlp2 = torch.nn.Linear(256, 32)
        # self.mlp3 = torch.nn.Linear(128, 8)




   
    def forward(self, im, q):
        x = self.image_conv_layers(im)
        x = x.flatten(start_dim=1)
        x = self.image_fc_layers(x)
        # x = self.image_fc_layers(nn.AvgPool2d(8, 8)(x.view(1, 1, -1)).view(-1))
        y, _ = self.lstm(self.embedding(q), (torch.zeros(2, 9, 1024).cuda(), torch.zeros(2, 9, 1024).cuda()))
        # z = torch.cat((y[:,-1,:], x),1)
        z = y[:,-1,:] * x
        z = self.mlp1(z)
        z = torch.tanh(z)
        # z = self.mlp2(z)
        # z = leaky(z)
        # z = self.mlp3(z)
        # z = leaky(z)
        return z
        
    def create_conv_layer(self, number_of_layers, in_size, out_size):
        modules = []
        modules.append(nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1))

        for i in range(number_of_layers - 1):
            modules.append(nn.ReLU())

            modules.append(nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1))

        modules.append(nn.ReLU())
        # modules.append(nn.Dropout(0.5))

        modules.append(nn.MaxPool2d(kernel_size=2, stride=2))        
        return nn.Sequential(*modules)

    def create_fc_layer(self, input_size, output_size):
        return nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            # nn.Dropout(0.5)
        )