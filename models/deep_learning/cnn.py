from torch import nn, autograd

# 可以根据最后输出的大小，最后接一些别的层来完成分类等操作
# CNNExtractor完成CNN前面的编码部分。

class CNNExtractor(nn.Module):
    def __init__(self, input_channel=3):
        super().__init__()

        self.layer1 = nn.Sequential(
                        nn.Conv2d(input_channel,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=0),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU(),
                        nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())
        self.layer4 = nn.Sequential(
                        nn.Conv2d(64,64,kernel_size=3,padding=1),
                        nn.BatchNorm2d(64, momentum=1, affine=True),
                        nn.ReLU())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        return out

class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12544, 10)
        )

    def flatten(self, x):
        return x.view(x.shape[0], -1)

    def forward(self, x):
        return self.flatten(self.classifier(x))



class GuidedReLU(autograd.Function):

    def forward(self, input):
        # 保存forward的数值，backward的时候使用
        self.save_for_backward(input)
        # 截断，也就是ReLu
        output = input.clamp(min=0)
        return output

    def backward(self, grad_output):
        input = self.saved_tensors[0]
        guided_grad = grad_output.clone()
        # 克隆梯度之后，截断梯度（根据原本梯度和保存的forward数值）
        # 根据参考文献 STRIVING FOR SIMPLICITY: THE ALL CONVOLUTIONAL NET
        # 将梯度中对应梯度小于0和输入小于0的位置设置为0。
        guided_grad[guided_grad < 0] = 0
        guided_grad[input<0] = 0
        return guided_grad


class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        features_module = CNNExtractor(3)
        self.features = nn.ModuleList(list(features_module.children()))
        self.classifier = CNNClassifier()

        self.gradient = []
        self.last_conv = None

    def forward(self, x, guided=False):
        for i, l in enumerate(self.features):
            # 由于模型为多个sequential组合，所以此处修改为l[-1]，判断是否为ReLU层
            if guided and l[-1].__class__.__name__ == 'ReLU':
                l = nn.Sequential(*l[:-1])
                guided_func = GuidedReLU()

            x = l(x)
            if guided and l[-1].__class__.__name__ == 'ReLU':
                x = guided_func(x)

            #store the gradient and output for the last convolutional layer
            if i == 3:
                x.retain_grad()
                x.register_hook(lambda grad: self.gradient.append(grad))
                self.last_conv = x

        x = self.classifier(x.view(x.size(0), -1))
        return x