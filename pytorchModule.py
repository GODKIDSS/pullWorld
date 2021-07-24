
#                        .::::.
#                      .::::::::.
#                     :::::::::::
#                  ..:::::::::::'
#               '::::::::::::'
#                 .::::::::::
#            '::::::::::::::..
#                 ..::::::::::::.
#               ``::::::::::::::::
#                ::::``:::::::::'        .:::.
#               ::::'   ':::::'       .::::::::.
#             .::::'      ::::     .:::::::'::::.
#            .:::'       :::::  .:::::::::' ':::::.
#           .::'        :::::.:::::::::'      ':::::.
#          .::'         ::::::::::::::'         ``::::.
#      ...:::           ::::::::::::'              ``::.
#     ```` ':.          ':::::::::'                  ::::..
#                        '.:::::'                    ':'````..
#                     美女保佑 永无BUG


import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

class fModel(nn.Module):
    def __init__(self, layerList=[128, 64, 16, 10]):
        super().__init__()
        self.layer1 = nn.Linear(layerList[0], layerList[1])
        self.layer2 = nn.Linear(layerList[1], layerList[2])
        self.layer3 = nn.Linear(layerList[2], layerList[3])
    def forward(self, input):
        out = self.layer1(input)
        out = self.layer2(out)
        out = self.layer3(out)
        return out

def trainModel(model, train_loader, optimizer, lossFunc):
    size = len(train_loader)
    loss_total = 0
    for batch_idx, (img, label) in enumerate(train_loader):
        print('\rtraining process{:.2%}'.format(batch_idx/size),end='')
        optimizer.zero_grad() # 清空过往梯度，否则不同batch的loss就叠加了
        img, label = img.to(device), label.to(device)
        img = img.view(img.size(0), -1) # 平铺
        model_out = model(img)
        loss = lossFunc(model_out, label)
        loss_total += loss.item()
        # optimizer.zero_grad() # 清空过往梯度，放在这里也可以
        loss.backward() # 重新计算梯度
        optimizer.step() # 梯度反向传播
        # 答疑：为什么loss与optimizer没关联起来，optimizer还能根据loss的梯度反向传播？
        # 回答：因为loss计算的是关于model的权重的函数，而optimizer与model关联，所以可以更新
    return loss_total/batch_idx

if __name__ == '__main__':
    # 定义超参
    EPOCH_NUM = 5
    BATCH_SIZE = 256
    LEARNING_RATE = 0.01
    # 加载数据集
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 定义模型，优化器与损失函数
    model = fModel([28*28,64,16,10])
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lossFunc = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model = model.to(device)
    # 训练
    for epoch in range(EPOCH_NUM):
        loss = trainModel(model, train_loader, optimizer, lossFunc)
        print('\nepoch',epoch,' ',loss)

