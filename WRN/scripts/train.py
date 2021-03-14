"""
Author: Zhou Chen
Date: 2020/4/16
Desc: 训练resnet50-2
"""
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torch.utils import data

from model import WRN50_2

# GPU配置
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    cudnn.benchmark = True
device = "cuda:{}".format(0) if torch.cuda.is_available() else "cpu"


batch_size = 128
img_size = 224
epochs = 100


def accuracy(output, label):
    pred = torch.argmax(output, dim=-1)
    correct = float(torch.sum(pred.eq(label)))
    return correct / output.size(0)


def train(train_loader, model, criterion, optimizer):
    model.train()
    losses = 0.0
    step = 0
    for step, (data, label) in enumerate(train_loader):
        label = label.to(device)
        data = data.to(device)
        output = model(data)

        loss = criterion(output, label)
        acc = accuracy(output.data, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print("training step: {}, loss: {}, acc: {}".format(step, loss.item() / data.size(0), acc))
        losses += loss.item() / data.size(0)

    return losses / (step + 1)


def validate(valid_loader, model, criterion):
    model.eval()
    losses = 0.0
    epoch_accuracy = 0.0
    step = 0
    for step, (data, label) in enumerate(valid_loader):
        label = label.to(device)
        data = data.to(device)

        with torch.no_grad():
            output = model(data)
        loss = criterion(output, label)
        epoch_accuracy = accuracy(output.data, label)
        if step % 50 == 0:
            print("validation step: {}, loss: {}, acc: {}".format(step, loss.data.item() / data.size(0), epoch_accuracy))
        losses += loss.item() / data.size(0)
    return epoch_accuracy, losses / (step + 1)


def main(augment=True):
    # 数据准备
    # 是否增广，采用原论文的增广方法
    transform_train = transforms.Compose([
        transforms.Resize(240),
        transforms.RandomCrop(224, padding=4, padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    transform_valid = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = data.DataLoader(
        datasets.ImageFolder('../data/Caltech101/train/', transform=transform_train),
        batch_size=batch_size,
        shuffle=True)
    valid_loader = data.DataLoader(
        datasets.ImageFolder('../data/Caltech101/valid/', transform=transform_valid),
        batch_size=batch_size,
        shuffle=True)
    print(train_loader.dataset.classes)
    print("the number of train images: {}".format(len(train_loader.dataset)))
    print("the number of valid images: {}".format(len(valid_loader.dataset)))

    # 构建模型
    model = WRN50_2()
    model.to(device)

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
    # 转移到GPU
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True, weight_decay=5e-4)

    his = {
        'train_loss': [],
        'valid_loss': []
    }

    best_acc = 0.0
    for epoch in range(epochs):
        train_loss = train(train_loader, model, criterion, optimizer)
        acc, valid_loss = validate(valid_loader, model, criterion)
        his['train_loss'].append(train_loss)
        his['valid_loss'].append(valid_loss)

        is_best = acc > best_acc
        if is_best:
            # 本地保存更好的模型参数
            best_acc = acc
            torch.save({'state_dict': model.state_dict(), }, 'weights.pth')

    with open('his.pkl', 'wb') as f:
        pickle.dump(his, f)


if __name__ == '__main__':
    main()
