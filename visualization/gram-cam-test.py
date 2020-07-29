import torch
from torch import nn

import sys
sys.path.append("C:/Users/KyoDante/Desktop/Torch_Signal_package/")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=8, affine=False),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16, affine=False),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32, affine=False),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(in_features=1568, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=10),
            # nn.Softmax(),
        )
        
        self.gradient = []
        self.last_conv = None

    # def forward(self, input):
    #     return self.layers(input)
    
    def forward(self, x, guided=False):
        for i, l in enumerate(self.layers):
            x = l(x)
            
            if i == 10:
                print(l)
                x.retain_grad()
                x.register_hook(lambda grad: self.gradient.append(grad))
                self.last_conv = x

        return x


if __name__ == "__main__":
    from utils.pytorch2mobile.torch2torchMobile import load_model, save_model
    # save_model(wd_model, "./hello.pt")
    wd_model = torch.jit.load("./wd_cpu.pt")
    wd_model_new = CNN()
    print(wd_model_new(torch.rand((1,3,28,28))))
    from visualization.grad_cam import grad_cam
    import PIL.Image
    import torchvision
    img = torchvision.transforms.ToTensor()(PIL.Image.open("D:/AcouDigits/journalExtension/AcouDigits_图片数据/压缩后的图片数据/第一次数字实验/WD_28/2/2.jpg"))
    print(img)
    grad_cam(wd_model_new, img.unsqueeze(0))
    
    # for idx, param in enumerate(wd_model.parameters()):
    #     param = list(wd_model.parameters())[idx]
    # print(list(wd_model.parameters())[0])
    # print(list(wd_model_new.parameters())[0])
    
    # torch.jit.script(wd_model).load_state_dict(torch.jit.load("./wd_cpu.pt"))
    # wd_model_jit = torch.jit.trace(wd_model, torch.rand((1,3,28,28)))
    # wd_model_jit.load_state_dict(torch.jit.load("./wd_cpu.pt"))
    # load_model(wd_model, "C:/Users/KyoDante/Desktop/AcouDigits/wd_cpu.pt")
    # torch.load("C:/Users/KyoDante/Desktop/AcouDigits/wd_cpu.pt", map_location='cpu')
    # print(wd_model)