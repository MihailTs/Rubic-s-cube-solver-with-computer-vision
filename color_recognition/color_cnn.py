import torch.nn as nn
import torch.nn.functional as F

class ColorCNN(nn.Module):

    def __init__(self,
                 num_classes,
                 pooling='max',
                 kernel_size=3,
                 activation_function='relu'):
        super().__init__()

        self.kernel_size = kernel_size
        if kernel_size == 3:
           self.padding = 1
        elif kernel_size == 5:
           self.padding = 2
        else:
           raise ValueError('kernel_size should either be 3 or 5')

        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size, padding=self.padding)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, padding=self.padding)

        if pooling == 'max_pooling':
            self.pool = nn.MaxPool2d(2, 2)
        elif pooling == 'average_pooling':
            self.pool = nn.AvgPool2d(2, 2)
        else:
            raise ValueError("Illegal pooling method")

        if activation_function != 'relu' and activation_function != 'elu' and activation_function != 'sigmoid' and activation_function != 'tanh':
            raise ValueError("Illegal activation function")
        self.activation_function = activation_function

        self.fc1 = nn.Linear(32 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        if self.activation_function == 'relu':
          x = self.pool(F.relu(self.conv1(x)))
          x = self.pool(F.relu(self.conv2(x)))
        elif self.activation_function == 'elu':
          x = self.pool(F.elu(self.conv1(x)))
          x = self.pool(F.elu(self.conv2(x)))
        elif self.activation_function == 'sigmoid':
          x = self.pool(F.sigmoid(self.conv1(x)))
          x = self.pool(F.sigmoid(self.conv2(x)))
        elif self.activation_function == 'tanh':
          x = self.pool(F.tanh(self.conv1(x)))
          x = self.pool(F.tanh(self.conv2(x)))

        x = x.view(x.size(0), -1)
        if self.activation_function == 'relu':
          x = F.relu(self.fc1(x))
        elif self.activation_function == 'elu':
          x = F.elu(self.fc1(x))
        elif self.activation_function == 'sigmoid':
          x = F.sigmoid(self.fc1(x))
        elif self.activation_function == 'tanh':
          x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return x
