from itertools import cycle

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import time
import os
from tqdm import tqdm
from model_resnet import resnet34, resnet50, resnet101, resnext101_32x8d
from model_vgg import vgg
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

import setproctitle

setproctitle.setproctitle("lihp_resnet3450")


# 定义数据读入
def Load_Image_Information(path):
    # 图像存储路径
    # image_Root_Dir = r'../shangHeDou_a'
    image_Root_Dir = r'../datasets/rawshdgzma2000'
    # 获取图像的路径
    iamge_Dir = os.path.join(image_Root_Dir, path)
    # 以RGB格式打开图像
    # Pytorch DataLoader就是使用PIL所读取的图像格式
    # 建议就用这种方法读取图像，当读入灰度图像时convert('')
    return Image.open(iamge_Dir).convert('RGB')


# 定义自己数据集的数据读入类
class my_Data_Set(nn.Module):
    def __init__(self, txt, transform=None, target_transform=None, loader=None):
        super(my_Data_Set, self).__init__()
        # 打开存储图像名与标签的txt文件
        fp = open(txt, 'r')
        images = []
        labels = []
        # 将图像名和图像标签对应存储起来
        for line in fp:
            line.strip('\n')
            line.rstrip()
            information = line.split()
            images.append(information[0])
            # 将标签信息由str类型转换为float类型
            labels.append(int(information[1]))
        self.images = images
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    # 重写这个函数用来进行图像数据的读取
    def __getitem__(self, item):
        # 获取图像名和标签
        imageName = self.images[item]
        label = self.labels[item]
        # 读入图像信息
        image = self.loader(imageName)
        # 处理图像数据
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    # 重写这个函数，来看数据集中含有多少数据
    def __len__(self):
        return len(self.images)


# 训练与验证网络（所有层都参加训练）
def train_model(model, criterion, optimizer, task_name, sep, model_name, pretrainedtf, repeatTimes, task_class_num,
                save_path, posc, isAug, bso, lro, num_epochs=400):
    Sigmoid_fun = nn.Sigmoid()
    since = time.time()
    best_f1 = 0.0
    model_epoch_choice = 0

    metric_lbs = [i for i in range(task_class_num)]
    # save_path = "../r220426/"
    pth_path = save_path + 'pth/'
    png_txt_path = save_path + 'opt/'
    if (not os.path.exists(pth_path)):
        os.mkdir(pth_path)
    if (not os.path.exists(png_txt_path)):
        os.mkdir(png_txt_path)

    train_loss_list = []
    val_loss_list = []
    res_list = []
    res_preds = []
    res_trues = []
    res_pscore = []

    print("using {} images for training, {} images for validation.".format(dataset_sizes['train'],
                                                                           dataset_sizes['val']))

    tname = str(task_name) + '_' + str(model_name) + '_' + str(sep) + '_' + str(pretrainedtf) + '_' + str(repeatTimes)+ '_' + isAug + '_' + str(bso)+ '_' + str(lro)
    best_save_path = pth_path + tname + '_best.pth'
    final_save_path = pth_path + tname + '_final.pth'

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每训练一个epoch，验证一下网络模型
        for phase in ['train', 'val']:
            running_loss = 0.0
            running_corrects = 0.0
            preds_list = []
            trues_list = []
            score_list = []

            if phase == 'train':
                # 学习率更新方式
                # scheduler.step()
                #  调用模型训练
                model.train()
                # 依次获取所有图像，参与模型训练或测试
                for data in dataloaders[phase]:
                    # 获取输入
                    inputs, labels = data
                    # 判断是否使用gpu
                    if use_gpu:
                        inputs = inputs.cuda()
                        labels = labels.cuda()

                    # inputs, labels = Variable(inputs), Variable(labels)

                    # 梯度清零
                    optimizer.zero_grad()

                    loss = 0
                    # 网络前向运行
                    if model_name == 'googlenet':
                        logits, aux_logits2, aux_logits1 = model(inputs)
                        loss0 = criterion(logits, labels)
                        loss1 = criterion(aux_logits1, labels)
                        loss2 = criterion(aux_logits2, labels)
                        loss = loss0 + loss1 * 0.3 + loss2 * 0.3
                    elif model_name == 'inceptionv3':
                        logits, aux_logits = model(inputs)
                        loss0 = criterion(logits, labels)
                        loss1 = criterion(aux_logits, labels)
                        loss = loss0 + loss1 * 0.4
                    else:
                        # 计算Loss值
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(Sigmoid_fun(outputs), labels)
                        preds_list.extend(preds.detach().cpu().numpy())
                        trues_list.extend(labels.detach().cpu().numpy())

                    # 反传梯度
                    loss.backward()
                    # 更新权重
                    optimizer.step()
                    # 计算一个epoch的loss值
                    running_loss += loss.item() * inputs.size(0)


            else:
                # 取消验证阶段的梯度
                with torch.no_grad():
                    # 调用模型测试
                    model.eval()
                    # 依次获取所有图像，参与模型训练或测试
                    for data in dataloaders[phase]:
                        # 获取输入
                        inputs, labels = data
                        # 判断是否使用gpu
                        if use_gpu:
                            inputs = inputs.cuda()
                            labels = labels.cuda()

                        # inputs, labels = Variable(inputs), Variable(labels)

                        # 网络前向运行
                        outputs = model(inputs)
                        _, preds = torch.max(outputs.data, 1)

                        for pred in range(len(preds)):
                            score_list.append(Sigmoid_fun(outputs.data[pred][posc]).detach().cpu().numpy())
                        # 计算Loss值
                        loss = criterion(Sigmoid_fun(outputs), labels)
                        # 计算一个epoch的loss值
                        running_loss += loss.item() * inputs.size(0)
                        # 计算一个epoch的准确率
                        running_corrects += torch.sum(preds == labels.data)

                        preds_list.extend(preds.detach().cpu().numpy())
                        trues_list.extend(labels.detach().cpu().numpy())

            # 计算Loss和准确率的均值
            epoch_loss = running_loss / dataset_sizes[phase]
            if phase == 'train':
                train_loss_list.append(epoch_loss)
            else:
                val_loss_list.append(epoch_loss)

            sklearn_accuracy = 0
            sklearn_precision = 0
            sklearn_recall = 0
            sklearn_f1 = 0
            if model_name in ['googlenet', 'inceptionv3'] and phase == 'train':
                continue
            else:
                # epoch_acc = float(running_corrects) / dataset_sizes[phase]
                sklearn_accuracy = accuracy_score(trues_list, preds_list)
                sklearn_precision = precision_score(trues_list, preds_list, average='micro')
                sklearn_recall = recall_score(trues_list, preds_list, average='micro')
                sklearn_f1 = f1_score(trues_list, preds_list, labels=metric_lbs, average='micro')

            print('Epoch: {} Phase: {} Loss: {:.4f} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}'.format(
                epoch, phase, epoch_loss, sklearn_accuracy, sklearn_precision, sklearn_recall, sklearn_f1))

            if phase == 'val' and sklearn_f1 > best_f1:
                best_f1 = sklearn_f1
                torch.save(model.state_dict(), best_save_path)
                model_epoch_choice = epoch

            if epoch == num_epochs - 1 and phase == 'train':
                torch.save(model.state_dict(), final_save_path)

            if epoch == num_epochs - 1 and phase == 'val':
                res_list.append(sklearn_accuracy)
                res_list.append(sklearn_precision)
                res_list.append(sklearn_recall)
                res_list.append(sklearn_f1)
                res_list.append(model_epoch_choice)

                res_preds.extend(preds_list)
                res_trues.extend(trues_list)
                res_pscore.extend(score_list)

    print("model_epoch_choice: ", model_epoch_choice)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    train_loss_list2 = pd.DataFrame(train_loss_list)
    val_loss_list2 = pd.DataFrame(val_loss_list)
    res_list2 = pd.DataFrame(res_list)
    res_preds2 = pd.DataFrame(res_preds)
    res_trues2 = pd.DataFrame(res_trues)
    res_pscore2 = pd.DataFrame(res_pscore)

    tlosspath = png_txt_path + tname + '_trainloss.txt'
    vlosspath = png_txt_path + tname + '_valloss.txt'
    respath = png_txt_path + tname + '_res.txt'
    resppath = png_txt_path + tname + '_resp.txt'
    restpath = png_txt_path + tname + '_rest.txt'
    respspath = png_txt_path + tname + '_resps.txt'
    rocpath = png_txt_path + tname + '_roc.png'
    lossfigpath = png_txt_path + tname + '_loss.png'

    repopath = png_txt_path + tname + '_repo.txt'
    cmpath = png_txt_path + tname + '_cm.png'

    train_loss_list2.to_csv(tlosspath, header=None, sep=' ', index=False)
    val_loss_list2.to_csv(vlosspath, header=None, sep=' ', index=False)
    res_list2.to_csv(respath, header=None, sep=' ', index=False)
    res_preds2.to_csv(resppath, header=None, sep=' ', index=False)
    res_trues2.to_csv(restpath, header=None, sep=' ', index=False)
    res_pscore2.to_csv(respspath, header=None, sep=' ', index=False)

    classfication_rep = classification_report(res_trues, res_preds, digits=4)
    print(classfication_rep)
    classfication_rep_txt = open(repopath, 'w')
    classfication_rep_txt.write(classfication_rep)
    classfication_rep_txt.close()


if __name__ == '__main__':
    # 是否使用gpu运算
    use_gpu = torch.cuda.is_available()
    # 任务列表
    models_lists = ['resnet50']
    every_task_runtime = 1
    task_names = ["sto_vt"]
    tasks_size = len(task_names)

    seps = ["73"]
    task_class_num = [2]
    poscs = [1]

    lrlist = [0.005,  0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    #lrlist = [0.005, 0.001, 0.0005]
    #lrlist = [0.0001, 0.00005, 0.00001]
    bslist = [8, 16, 32, 64]
    #bslist = [8, 16]
    #bslist = [32, 64]
    auglist = ["aug", "noaug"]
    # task_class_num = [3, 2]
    save_path = "../sto1023/"
    if (not os.path.exists(save_path)):
        os.mkdir(save_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))


    for i in range(tasks_size):
        task_name = task_names[i]

        for m in range(every_task_runtime):

            for j in range(len(seps)):
                sep = seps[j]
                train_path = r'../trainSepe/train_' + task_name + '_' + sep + '.txt'
                val_path = r'../trainSepe/val_' + task_name + '_' + sep + '.txt'
                assert os.path.exists(train_path), "file {} does not exist.".format(train_path)
                assert os.path.exists(val_path), "file {} does not exist.".format(val_path)

                for isAug in auglist:
                    data_transforms = {
                        'train': transforms.Compose([
                            transforms.Resize(224),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ]),
                        'val': transforms.Compose([
                            transforms.Resize(224),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ]),
                    }
                    if isAug == "aug":
                        data_transforms = {
                            'train': transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomRotation(10),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ]),
                            'val': transforms.Compose([
                                transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ]),
                        }

                    train_Data = my_Data_Set(train_path, transform=data_transforms['train'],
                                             loader=Load_Image_Information)
                    val_Data = my_Data_Set(val_path, transform=data_transforms['val'],
                                           loader=Load_Image_Information)

                    for pp in range(len(bslist)):
                        train_DataLoader = DataLoader(train_Data, batch_size=bslist[pp], shuffle=True)
                        val_DataLoader = DataLoader(val_Data, batch_size=bslist[pp])
                        dataloaders = {'train': train_DataLoader, 'val': val_DataLoader}
                        # 读取数据集大小
                        dataset_sizes = {'train': train_Data.__len__(), 'val': val_Data.__len__()}

                        for qq in range(len(lrlist)):


                            for k in range(len(models_lists)):
                                model = resnet34()
                                model_name = models_lists[k]
                                model_weight_path = "../pretrained/" + model_name + "-pre.pth"
                                assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)


                                for l in range(2):
                                    if model_name == 'resnet50':
                                        model = resnet50()

                                    # 是否预训练
                                    if l == 1:
                                        model.load_state_dict(torch.load(model_weight_path, map_location="cpu"))

                                    in_channel = model.fc.in_features
                                    model.fc = nn.Linear(in_channel, task_class_num[i])

                                    model.to(device)

                                    # 定义损失函数
                                    criterion = nn.CrossEntropyLoss()



                                    # construct an optimizer
                                    params = [p for p in model.parameters() if p.requires_grad]
                                    optimizer = torch.optim.Adam(params, lr=lrlist[qq])


                                    train_model(model, criterion, optimizer, task_name, sep, model_name, l, m, task_class_num[i],
                                                save_path, poscs[i], isAug, bslist[pp], lrlist[qq], num_epochs=50)
                                    print(
                                        'Finished Training ' + task_name + ' ' + sep + ' ' + model_name + ' ' + str(l) + ' ' + str(
                                            m) + ' ' + isAug + ' ' + str(bslist[pp]) + str(lrlist[qq]))