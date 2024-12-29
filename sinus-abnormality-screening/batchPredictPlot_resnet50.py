from itertools import cycle

import PIL.Image
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
import torchvision
import setproctitle
setproctitle.setproctitle("lihp_batchP")
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# 定义数据读入
def Load_Image_Information(path):
    # 图像存储路径
    image_Root_Dir = r'../datasets/shdgzma2000'
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

def get_loss_fig(train_loss_list, val_loss_list, lossfigpath, num_epochs):
    plt.clf()
    x = [i for i in range(num_epochs)]

    fig = plt.figure(1)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(x, train_loss_list, c='b', linewidth=4.0, linestyle='solid', label='train_loss')
    plt.plot(x, val_loss_list, c='r', linewidth=4.0, linestyle='solid', label='val_loss')
    plt.grid(True)  # 显示网格

    plt.legend()#显示旁注#注意：不会显示后来再定义的旁注
    # fig.show()
    fig.savefig(lossfigpath)


def get_pr_ap(pa_precision, pa_recall,average_precision, papath):
    plt.clf()
    plt.figure("P-R Curve")
    plt.title('Precision/Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.plot(pa_recall, pa_precision,label='PR curve (area = {0:0.3f})'.format(average_precision))
    plt.legend(loc='best')
    plt.savefig(papath)

def get_roc_auc(trues, preds, labels, rocpath):
    plt.clf()
    nb_classes = len(labels)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(nb_classes):
        fpr[i], tpr[i], _ = roc_curve(trues[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(trues.ravel(), preds.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(nb_classes)]))
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(nb_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    # Finally average it and compute AUC
    mean_tpr /= nb_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)
    plt.plot(fpr["macro"], tpr["macro"], label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(nb_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.savefig(rocpath)

def get_roc_auc2(trues, preds, rocpath):
    plt.clf()
    fpr, tpr, thresholds_keras = roc_curve(trues, preds)
    roc_auc = auc(fpr, tpr)
    print("AUC : ", roc_auc)
    plt.figure()
    plt.plot([0,1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='ROC curve (area = {:.3f})'.format(roc_auc),clip_on=False)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig(rocpath)

def plot_confusion_matrix(conf_matrix, labels, save_path):
    plt.clf()
    plt.imshow(conf_matrix, cmap=plt.cm.Greens)
    indices = range(conf_matrix.shape[0])
    plt.xticks(indices, labels)
    plt.yticks(indices, labels)
    plt.colorbar()
    plt.xlabel('y_true')
    plt.ylabel('y_pred')
    # 显示数据
    for first_index in range(conf_matrix.shape[0]):
        for second_index in range(conf_matrix.shape[1]):
            plt.text(first_index, second_index, conf_matrix[first_index, second_index])
    plt.savefig(save_path)

#获取热力图
def gt_cam(model,target_layers,inputs_tensor,img):
    cam = GradCAM(model=model,target_layers=target_layers,use_cuda=True)
    #targets = [ClassifierOutputTarget(0)]
    grayscale_cam = cam(input_tensor=inputs_tensor)
    grayscale_cam = grayscale_cam[0,:]
    visualization = show_cam_on_image(img,grayscale_cam,use_rgb=True)
    # cv2.imwrite('../gt_camm/'+tname+'_'+f'{label}'+'_'+ f'{flag}.png',cv2.cvtColor(visualization,cv2.COLOR_BGR2RGB))
    return visualization

# 定义验证阶段
def val(model, criterion, task_class_num, posc, tname, phase='test'):
    # 创建文件夹
    save_path = f"../sto1110tpbest/{phase}"
    png_txt_path = save_path + f'/opt_{phase}/'
    if (not os.path.exists(save_path)):
        os.makedirs(save_path)
    if (not os.path.exists(png_txt_path)):
        os.makedirs(png_txt_path)
    if (not os.path.exists(f"{save_path}/grad_cam/{tname}")):
        os.makedirs(f"{save_path}/grad_cam/{tname}")
    if (not os.path.exists(f"{save_path}/visual/{tname}/TP")):
        os.makedirs(f"{save_path}/visual/{tname}/TP")
    if (not os.path.exists(f"{save_path}/visual/{tname}/TN")):
        os.makedirs(f"{save_path}/visual/{tname}/TN")
    if (not os.path.exists(f"{save_path}/visual/{tname}/FP")):
        os.makedirs(f"{save_path}/visual/{tname}/FP")
    if (not os.path.exists(f"{save_path}/visual/{tname}/FN")):
        os.makedirs(f"{save_path}/visual/{tname}/FN")

    # 模型验证
    model.eval()
    # 指定不保存梯度
    with torch.no_grad():

        Sigmoid_fun = nn.Sigmoid()
        metric_lbs = [i for i in range(task_class_num)]
        res_list = []
        print("using {} images for validation.".format(dataset_sizes[phase]))

        # 统计Loss值与准确率
        running_loss = 0.0
        running_corrects = 0

        preds_list = []
        trues_list = []
        score_list = []

        for batch,data in enumerate(dataloaders[phase]):
            inputs, labels = data
             #保存每个原始batch图片
            # batch_img = torchvision.utils.make_grid(inputs,nrow=4,padding=0,normalize=True)
            # batch_img = torchvision.transforms.ToPILImage()(batch_img)
            # batch_img.save(f"{save_path}/grad_cam/{tname}/original_batch_{batch}.png")

            # 判断是否使用gpu
            if use_gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            # 模型前向运行
            outputs = model(inputs)
            # 获取预测结果
            _, preds = torch.max(outputs.data, 1)
           
            for pred in range(len(preds)):
                score_list.append(Sigmoid_fun(outputs.data[pred][posc]).detach().cpu().numpy())
            # 计算Loss值
            loss = criterion(Sigmoid_fun(outputs), labels)
            # 统计Loss值和准确率
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            true_batch = labels.detach().cpu().numpy()
            pred_batch = preds.detach().cpu().numpy()
            preds_list.extend(pred_batch)
            trues_list.extend(true_batch)


            #卷积可视化
            torch.set_grad_enabled(True)
            target_layers = [model.layer4[-1]]
            for w,[label,input] in enumerate(zip(labels,inputs)):
                img = torchvision.utils.make_grid(input,nrow=1,padding=0,normalize=True)
                origin_img = torchvision.transforms.ToPILImage()(img)
                img = np.asanyarray(torchvision.transforms.ToPILImage()(img),dtype=np.float32)/255
                GT_IMG = gt_cam(model,target_layers,torch.unsqueeze(input,dim=0),img)
                GT_IMG = Image.fromarray(GT_IMG)
                origin_img.save(f"{save_path}/grad_cam/{tname}/original_batch_{batch}_NO.{w}.png")
                GT_IMG.save(f"{save_path}/grad_cam/{tname}/gt_batch_{batch}_NO.{w}.png")

                # 判断TP TN FN FP
                if true_batch[w] == 1:
                    if true_batch[w] == pred_batch[w]:
                        origin_img.save(f"{save_path}/visual/{tname}/TP/original_batch_{batch}_NO.{w}_true{true_batch[w]}_pred{pred_batch[w]}.png")
                    if true_batch[w] != pred_batch[w]:
                        origin_img.save(f"{save_path}/visual/{tname}/FN/original_batch_{batch}_NO.{w}_true{true_batch[w]}_pred{pred_batch[w]}.png")
                if true_batch[w] == 0:
                    if true_batch[w] == pred_batch[w]:
                        origin_img.save(f"{save_path}/visual/{tname}/TN/original_batch_{batch}_NO.{w}_true{true_batch[w]}_pred{pred_batch[w]}.png")
                    if true_batch[w] != pred_batch[w]:
                        origin_img.save(f"{save_path}/visual/{tname}/FP/original_batch_{batch}_NO.{w}_true{true_batch[w]}_pred{pred_batch[w]}.png")
            torch.set_grad_enabled(False)

        epoch_loss = running_loss / dataset_sizes[phase]
        sklearn_accuracy = accuracy_score(trues_list, preds_list)
        sklearn_precision = precision_score(trues_list, preds_list, average='micro')
        sklearn_recall = recall_score(trues_list, preds_list, average='micro')
        sklearn_f1 = f1_score(trues_list, preds_list, labels=metric_lbs, average='micro')


        print('Loss: {:.4f} accuracy:{:.4f} precision:{:.4f} recall:{:.4f} f1:{:.4f}'.format(epoch_loss, sklearn_accuracy, sklearn_precision, sklearn_recall, sklearn_f1))

        res_list.append(sklearn_accuracy)
        res_list.append(sklearn_precision)
        res_list.append(sklearn_recall)
        res_list.append(sklearn_f1)

        res_list2 = pd.DataFrame(res_list)
        res_preds2 = pd.DataFrame(preds_list)
        res_trues2 = pd.DataFrame(trues_list)
        res_pscore2 = pd.DataFrame(score_list)

        respath = png_txt_path + tname + '_res.txt'
        resppath = png_txt_path + tname + '_resp.txt'
        res_prt_path = png_txt_path + tname + '_resprt.txt'
        restpath = png_txt_path + tname + '_rest.txt'
        respspath = png_txt_path + tname + '_resps.txt'

        rocpath = png_txt_path + tname + '_roc.png'
        repopath = png_txt_path + tname + '_repo.txt'
        cmpath = png_txt_path + tname + '_cm.png'

        res_list2.to_csv(respath, header=None, sep=' ', index=False)
        res_preds2.to_csv(resppath, header=None, sep=' ', index=False)
        res_trues2.to_csv(restpath, header=None, sep=' ', index=False)
        res_pscore2.to_csv(respspath, header=None, sep=' ', index=False)

        test_trues = label_binarize(trues_list, classes=metric_lbs)
        test_preds = label_binarize(preds_list, classes=metric_lbs)
        if task_class_num == 2:
            get_roc_auc2(test_trues, score_list, rocpath)
        else:
            get_roc_auc(test_trues, test_preds, metric_lbs, rocpath)

        if task_class_num == 2:
            average_precision = average_precision_score(trues_list, score_list)
            pa_precision, pa_recall, pa_thresholds = precision_recall_curve(trues_list, score_list)
            paap = average_precision_score(trues_list, score_list, average='macro', pos_label=posc, sample_weight=None)
            papath = png_txt_path + tname + '_pa_' + '{:.4f}.png'.format(paap)
            get_pr_ap(pa_precision, pa_recall,average_precision, papath)
            pa_f1 = (2*(pa_precision*pa_recall))/(pa_precision+pa_recall)
            with open(res_prt_path, 'w+') as f:
                f.write('{:<30}\t{:<30}\t{:<30}\t{:<30}\n'.format('pre','recall',"f1",'thre'))
                for i in range(len(pa_thresholds)):
                    f.write('{:<30}\t{:<30}\t{:<30}\t{:<30}\n'.format(str(pa_precision[i]),str(pa_recall[i]),str(pa_f1[i]),str(pa_thresholds[i])))
	
        classfication_rep = classification_report(trues_list, preds_list, digits=4)
        classfication_rep_txt = open(repopath, 'w')
        classfication_rep_txt.write(classfication_rep)
        classfication_rep_txt.close()

        conf_matrix = confusion_matrix(trues_list, preds_list, labels=metric_lbs)
        plot_confusion_matrix(conf_matrix, metric_lbs, cmpath)


if __name__ ==  '__main__':

    # 是否使用gpu运算
    use_gpu = torch.cuda.is_available()
    # 任务列表
    models_lists = ['resnet50']
    resizeSize = [224, 224, 224, 224]
    task_names = ["sto_vt"]
    tasks_size = len(task_names)
    seps = ["73"]
    task_class_num = [2]
    poscs = [1]
    lrlist = [0.0001]
    bslist = [32]
    auglist = ["aug", "noaug"]
    pth_path = "../sto0720/pth/"

    #画图字体大小设置
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = '16'

    # 定义数据的处理方式
    data_transforms = {
        'train': transforms.Compose([
            # 将图像进行缩放，缩放为256*256
            transforms.Resize(224),
            transforms.CenterCrop(224),
            # 在256*256的图像上随机裁剪出227*227大小的图像用于训练
            # transforms.RandomResizedCrop(227),
            # 图像用于翻转
            # transforms.RandomHorizontalFlip(),
            # 转换成tensor向量
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

        # 测试集需要中心裁剪，甚至不裁剪，直接缩放为224*224for，不需要翻转
        'val': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),

        'test': transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    # 导入Pytorch封装的AlexNet网络模型
    #model = models.alexnet(pretrained=True)
    # 获取最后一个全连接层的输入通道数
    #num_input = model.classifier[6].in_features
    # 获取全连接层的网络结构
    #feature_model = list(model.classifier.children())
    # 去掉原来的最后一层
    #feature_model.pop()
    # 添加上适用于自己数据集的全连接层
    # 260数据集的类别数
    #feature_model.append(nn.Linear(num_input, 260))
    # 仿照这里的方法，可以修改网络的结构，不仅可以修改最后一个全连接层
    # 还可以为网络添加新的层
    # 重新生成网络的后半部分
    #model.classifier = nn.Sequential(*feature_model)

    # 生成Pytorch所需的DataLoader数据输入格式
    for i in range(tasks_size):
        task_name = task_names[i]

        for j in range(len(seps)):
            sep = seps[j]
            train_path = r'../trainSepe/train_' + task_name + '_' + sep + '.txt'
            val_path = r'../trainSepe/val_' + task_name + '_' + sep + '.txt'
            test_path = r'../trainSepe/test_' + task_name + '_' + sep + '.txt'
            assert os.path.exists(train_path), "file {} does not exist.".format(train_path)
            assert os.path.exists(val_path), "file {} does not exist.".format(val_path)
            assert os.path.exists(test_path), "file {} does not exist.".format(test_path)

            for k in range(len(models_lists)):
                model = resnet34()
                model_name = models_lists[k]

                for l in range(1, 2):
                    for isAug in auglist:
                        for pp in range(len(bslist)):
                            for qq in range(len(lrlist)):
                                tname = str(task_name) + '_' + str(model_name) + '_' + str(sep) + '_' + str(l) + '_0_' + isAug + '_' + str(bslist[pp])+ '_' + str(lrlist[qq])

                                model_weight_path = pth_path + tname + '_best.pth'
                                assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
                                train_Data = my_Data_Set(train_path, transform=data_transforms['train'],
                                                         loader=Load_Image_Information)
                                val_Data = my_Data_Set(val_path, transform=data_transforms['val'],
                                                       loader=Load_Image_Information)
                                test_Data = my_Data_Set(test_path, transform=data_transforms['test'],
                                                       loader=Load_Image_Information)

                                train_DataLoader = DataLoader(train_Data, batch_size=16, shuffle=True)
                                val_DataLoader = DataLoader(val_Data, batch_size=16)
                                test_DataLoader = DataLoader(test_Data, batch_size=16)

                                dataloaders = {'train': train_DataLoader, 'val': val_DataLoader, 'test':test_DataLoader}
                                # 读取数据集大小
                                dataset_sizes = {'train': train_Data.__len__(), 'val': val_Data.__len__(), 'test': test_Data.__len__()}

                                if model_name == 'resnet50':
                                    model = resnet50()

                                in_channel = model.fc.in_features
                                model.fc = nn.Linear(in_channel, task_class_num[i])

                                model.load_state_dict(torch.load(model_weight_path, map_location="cpu"))
                                model.to(device)

                                # 定义损失函数
                                criterion = nn.CrossEntropyLoss()
                                flag = 0
                                val(model, criterion, task_class_num[i], poscs[i], tname)
                                print('Finished Training ' + tname)
