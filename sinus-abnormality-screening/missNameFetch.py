import os.path

work_dir = "..\\yolo_raw_json"
json_list = os.listdir(work_dir)
json_num = len(json_list)
name_list = []

for i in range(json_num):
    json_cur = json_list[i]
    new_filename = ""
    for j in range(len(json_cur)):
        if json_cur[j] >= '0' and json_cur[j] <= '9':
            new_filename = new_filename + json_cur[j]
        else:
            break

    if ("16P" in json_cur or "26P" in json_cur or "16p" in json_cur or "26p" in json_cur):
        new_filename = new_filename + '-'
        if ("16P" in json_cur or "16p" in json_cur):
            new_filename = new_filename + "16P"
        else:
            new_filename = new_filename + "26P"
        name_list.append(new_filename)

count = 0

for i in range(201, 1000):
    name1 = str(i) + "-16P"
    name2 = str(i) + "-26P"
    if name1 not in name_list:
        print(name1)
        count += 1
    if name2 not in name_list:
        print(name2)
        count += 1

print("miss num: ",  count)
print("Finished!")