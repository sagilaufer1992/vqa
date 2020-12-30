import torch
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

        self.embedding = nn.Embedding(1024, 16)
        self.lstm = nn.LSTM(16, 1024)

   
    def forward(self, im, q):
        # b = self.image_layer1(im)
        image_result = self.image_conv_layers(im)
        image_result = image_result.flatten(start_dim=1)
        image_result = self.image_fc_layers(image_result)
        question_result, hidden = self.lstm(self.embedding(q), (torch.zeros(1, 16, 1024), torch.zeros(1, 16, 1024)))
        
        print(question_result[:,-1,:].size())
        print(image_result.size())

        return image_result, question_result

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