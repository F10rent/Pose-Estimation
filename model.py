import torch
import torch.nn as nn

class CPM(nn.Module):
    def __init__(self):
        super(CPM, self).__init__()
        
        self.pre_conv_layers = nn.Sequential(
            nn.Conv2d(3, 128, 9, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(128, 128, 9, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(128, 128, 9, stride=1, padding=4),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(128, 32, 5, stride=1, padding=2),
            nn.ReLU(),
        )

        # Stage 1 layers
        self.stage1_conv_layers = nn.Sequential(
            nn.Conv2d(32, 512, 9, stride=1, padding=4),
            nn.ReLU(),
            nn.Conv2d(512, 512, 1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(512, 14, 1, stride=1, padding=0),
        )

        # Stages 2-4 shared layers
        self.Mconv_layers = nn.Sequential(
            nn.Conv2d(46, 128, 11, stride=1, padding=5),
            nn.ReLU(),
            nn.Conv2d(128, 128, 11, stride=1, padding=5),
            nn.ReLU(),
            nn.Conv2d(128, 128, 11, stride=1, padding=5),
            nn.ReLU(),
            nn.Conv2d(128, 128, 1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 14, 1, stride=1, padding=0),
        )

    def forward(self, x):
        image_pre = self.pre_conv_layers(x)
        
        # Stage 1
        stage1_out = self.stage1_conv_layers(image_pre)

        # Initialize list to store outputs for each stage
        stage_outputs = [stage1_out]

        # Stages 2-4
        concat_stage = torch.cat((image_pre, stage1_out), dim=1)
        for _ in range(3):
            stage_out = self.Mconv_layers(concat_stage)
            concat_stage = torch.cat((image_pre, stage_out), dim=1)
            stage_outputs.append(stage_out)

        return stage_outputs


