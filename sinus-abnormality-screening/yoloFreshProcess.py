# -*- coding: utf-8 -*-

# lib
# ==================================================
import os.path
import json
import cv2

from shutil import move, copy

# ==================================================

# config
# ==================================================
changeImageName = False  # Step 1: 文件更名及名字映射的存储开关
checkMissingData = False  # 查找000-dataUpperBound(<1000)号不存在数据的患者
dataLowerBound = 101
dataUpperBound = 1000
guanZhuangMian = True  # 对冠状面进行处理，默认为True，False即是对矢状面进行处理
processType = "guanZhuangMian"  # 影响后续映射文件及文件夹命名
if not guanZhuangMian:
    processType = "shiZhuangMian"
step1_work_dir = "..\\yolo_fresh"  # 原始待处理图像数据所在路径
step1_guanZhuangMian_output_dir = "..\\yolo_fresh_rename"  # 处理完毕冠状面图像输出路径
step1_shiZhuangMian_output_dir = ""  # 处理完毕矢状面图像输出路径
fileType = '.tif'  # 原始待处理图像数据格式

changeImageFormat = False  # Step 2: 文件格式更改开关
formatTo = ".jpg"  # 更改后的新格式
step2_work_dir = "..\\yolo_fresh_rename"  # 原始待处理图像数据所在路径
step2_output_dir = "..\\yolo_fresh_" + processType  # 处理完毕图像输出路径

cropImage = True  # Step 3: 初步裁剪图像开关
step3_work_dir = step2_output_dir
step3_output_dir = "..\\cropLeft_" + processType  # 处理完毕图像输出路径
# ==================================================

# Step 1: 文件更名及名字映射的存储
# ==================================================
if changeImageName:
    print("******************************")
    print("Step 1 start!")
    if (not os.path.exists(step1_guanZhuangMian_output_dir)):
        os.mkdir(step1_guanZhuangMian_output_dir)

    original_list = os.listdir(step1_work_dir)
    img_num = len(original_list)
    conflict = []

    for i in range(img_num):
        img_original = original_list[i]
        img_dir_original = os.path.join(step1_work_dir, img_original)
        new_filename = ""
        if img_original[-len(fileType):] == fileType:
            new_filetype = fileType
        else:
            new_filetype = '.json'

        for j in range(len(img_original)):
            if img_original[j] >= '0' and img_original[j] <= '9':
                new_filename = new_filename + img_original[j]
            else:
                break

        if ("16P" in img_original or "26P" in img_original or "16p" in img_original or "26p" in img_original):
            new_filename = new_filename + '-'
            if ("16P" in img_original or "16p" in img_original):
                new_filename = new_filename + "16P"
            else:
                new_filename = new_filename + "26P"

            cf = new_filename + new_filetype
            if cf in conflict:
                print(img_original)
                continue
            else:
                conflict.append(cf)

            img_dir_new = os.path.join(step1_guanZhuangMian_output_dir, new_filename + new_filetype)
            copy(img_dir_original, img_dir_new)

    print("Step 1 finished!")
    print("******************************")

# 查找000-dataUpperBound(<1000)号不存在数据的患者
if checkMissingData:
    data_list = os.listdir(step1_guanZhuangMian_output_dir)
    data_num = len(data_list)
    fig_name_list = []
    json_name_list = []

    for i in range(data_num):
        data_cur = data_list[i]
        new_filename = ""
        if data_cur[-len(fileType):] == fileType:
            new_filetype = fileType
        else:
            new_filetype = '.json'

        for j in range(len(data_cur)):
            if data_cur[j] >= '0' and data_cur[j] <= '9':
                new_filename = new_filename + data_cur[j]
            else:
                break

        if ("16P" in data_cur or "26P" in data_cur or "16p" in data_cur or "26p" in data_cur):
            new_filename = new_filename + '-'
            if ("16P" in data_cur or "16p" in data_cur):
                new_filename = new_filename + "16P"
            else:
                new_filename = new_filename + "26P"

            if data_cur[-len(fileType):] == fileType:
                fig_name_list.append(new_filename)
            else:
                json_name_list.append(new_filename)

    miss_fig_count = 0
    miss_json_count = 0

    for i in range(dataLowerBound, dataUpperBound):
        name1 = str(i) + "-16P"
        name2 = str(i) + "-26P"
        if name1 not in fig_name_list:
            print(name1)
            miss_fig_count += 1
        if name2 not in fig_name_list:
            print(name2)
            miss_fig_count += 1

    print("miss fig num: ",  miss_fig_count)

    for i in range(dataLowerBound, dataUpperBound):
        name1 = str(i) + "-16P"
        name2 = str(i) + "-26P"
        if name1 not in json_name_list:
            print(name1)
            miss_json_count += 1
        if name2 not in json_name_list:
            print(name2)
            miss_json_count += 1

    print("miss json num: ",  miss_json_count)

    print("checkMissingData Finished!")

# Step 2: 文件格式更改
if changeImageFormat:
    print("******************************")
    print("Step 2 start!")
    if (not os.path.exists(step2_output_dir)):
        os.mkdir(step2_output_dir)

    img_list = os.listdir(step2_work_dir)
    img_num = len(img_list)
    for i in range(img_num):
        img_name = img_list[i]
        img_dir = os.path.join(step2_work_dir, img_name)

        if img_name[-len(fileType):] == fileType:
            img = cv2.imread(img_dir, 1)
            img_dir_new = os.path.join(step2_output_dir, img_name[:-len(fileType)] + formatTo)
            cv2.imwrite(img_dir_new, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        else:
            img_dir_new = os.path.join(step2_output_dir, img_name)
            copy(img_dir, img_dir_new)

    print("Step 2 finished!")
    print("******************************")

# Step 3: 初步裁剪图像
if cropImage:
    print("******************************")
    print("Step 3 start!")
    if (not os.path.exists(step3_output_dir)):
        os.mkdir(step3_output_dir)

    img_list = os.listdir(step3_work_dir)
    img_num = len(img_list)
    for i in range(img_num):
        img_name = img_list[i]
        img_dir = os.path.join(step3_work_dir, img_name)

        if img_name[-len(formatTo):] == formatTo:
            img = cv2.imread(img_dir, 1)
            cropped = img[100:1356, 915:2560]
            cropped = cropped[0:900, 0:823]  # guanZhuangMian left
            img_dir_new = os.path.join(step3_output_dir, img_name)
            cv2.imwrite(img_dir_new, cropped, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        else:
            img_dir_new = os.path.join(step3_output_dir, img_name)
            copy(img_dir, img_dir_new)


    print("Step 3 finished!")
    print("******************************")