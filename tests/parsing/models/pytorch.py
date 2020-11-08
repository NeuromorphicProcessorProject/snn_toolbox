import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # The input_shape field is required by SNN toolbox.
        self.input_shape = (1, 28, 28)

        layers_trunk = [
            nn.Conv2d(1, 16, kernel_size=5, stride=2),
            # BatchNorm doesn't work with Keras==2.3.1 because for some reason
            # they put the batch-norm axis in a list.
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)]
        layers_branch1 = [
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU()]
        layers_branch2 = [
            nn.Conv2d(16, 8, kernel_size=1),
            nn.ReLU()]
        layers_head = [
            nn.Conv2d(40, 8, kernel_size=1),
            nn.ReLU()]
        layers_classifier = [
            nn.Dropout(1e-5),
            nn.Linear(288, 10),
            nn.Softmax(1)]
        self.trunk = nn.Sequential(*layers_trunk)
        self.branch1 = nn.Sequential(*layers_branch1)
        self.branch2 = nn.Sequential(*layers_branch2)
        self.head = nn.Sequential(*layers_head)
        self.classifier = nn.Sequential(*layers_classifier)

    def forward(self, x):
        x = self.trunk(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x = torch.cat([x1, x2], 1)
        x = self.head(x)
        x = x.view(-1, 288)  # Flatten
        x = self.classifier(x)
        return x
