import torch
import torch.nn as nn

class model(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, hidden_num, class_num, batch_size):
        super(model, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size = 3, padding = 1, bias = True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(mid_channel)
        self.conv2 = nn.Conv2d(in_channel, mid_channel, kernel_size = 3, padding = 1, bias = True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.activation = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(0.2)
        self.dense1 = nn.Linear(8024, hidden_num)
        self.dense2 = nn.Linear(hidden_num, class_num)
        self.softmax = nn.Softmax()
        self.need_init = [self.conv1, self.bn1, self.conv2, self.bn2, self.dense1, self.dense2]
        self.init_type = "normal"
        self.batch_size = batch_size

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.activation(bn1)
        pool1 = self.pool1(relu1)
        conv2 = self.conv2(pool1)
        bn2 = self.bn2(conv2)
        relu2 = self.activation(bn2)
        pool2 = self.pool2(relu2)

        dense_input = pool2.view(self.batch_size,-1)
        dense1 = self.dense1(dense_input)
        dense1 = self.activation(dense1)
        dropout1 = self.dropout1(dense1)
        dense2 = self.dense2(dropout1)
        out = self.softmax(dense2)
        return out

    def init_model(self):
        for layer in self.need_init:
            classname = layer.__class__.__name__
            if hasattr(layer, 'weight') and (classname.find("Conv")!=-1 or classname.find("Linear")!=-1):
                if self.init_type == "normal":
                    nn.init.normal_(layer.weight.data, 0.0, 0.02)
                elif self.init_type == 'xavier':
                    nn.init.xavier_normal_(layer.weight.data, gain = 0.02)
                elif self.init_type == 'kaiming':
                    nn.init.kaiming_normal_(layer.weight.data, a = 0 , mode = "fan_in")
                elif self.init_type == 'orthogonal':
                    nn.init.orthogonal_(layer.weight.data, gain = 0.02)
                if hasattr(layer, 'bias') and layer.bias is not None:
                    nn.init.constant_(layer.bias.data, 0.0)
            if classname.find("Batchnorm2d")!=-1:
                nn.init.normal_(layer.weight.data, 1.0, 0.02)
                nn.init.constant(layer.bias.data, 0.0)
        print('initialize network with %s' % self.init_type)







