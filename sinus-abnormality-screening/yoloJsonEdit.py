# -*- coding: utf-8 -*-

# lib
# ==================================================
import os.path
import json
from labelme import utils
import cv2
# ==================================================

dataLowerBound = 101
dataUpperBound = 1001
newx0 = 915
newy0 = 100

work_dir = "..\\cropLeft_guanZhuangMian"
json_output_dir = "..\\yolo_json_new"
if (not os.path.exists(json_output_dir)):
    os.mkdir(json_output_dir)

for i in range(dataLowerBound, dataUpperBound):
    for j in ["-16P", "-26P"]:
        js = str(i) + j + ".json"
        fig = str(i) + j + ".jpg"
        js_dir = os.path.join(work_dir, js)
        fig_dir = os.path.join(work_dir, fig)
        figr = cv2.imread(fig_dir)

        with open(js_dir, 'r', encoding='utf-8') as f1:
            jso = json.load(f1)
    
        jso["imageHeight"] = 900
        jso["imageWidth"] = 823
        jso["imagePath"] = fig
        jso["imageData"] = utils.img_arr_to_b64(figr).decode('utf-8')

        jso["shapes"][0]["points"][0][0] -= newx0
        jso["shapes"][0]["points"][1][0] -= newx0
        jso["shapes"][0]["points"][0][1] -= newy0
        jso["shapes"][0]["points"][1][1] -= newy0

        if jso["shapes"][0]["points"][0][0] < 0:
            jso["shapes"][0]["points"][0][0] = 0
        if jso["shapes"][0]["points"][0][1] < 0:
            jso["shapes"][0]["points"][0][1] = 0

        jsn = json.dumps(jso, ensure_ascii=False, sort_keys=True, indent=4, separators=(',', ': '))
        jsn_dir = os.path.join(json_output_dir, js)
        with open(jsn_dir, 'w', encoding='utf-8') as f2:
            f2.write(jsn)

