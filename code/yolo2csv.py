import os
import csv


def process_txt_to_csv(image_file,txt_folder, output_csv):
    # 获取文件夹名称作为image_id
    # image_id = os.path.basename(os.path.normpath(txt_folder))
    # 准备CSV文件
    with open(output_csv, 'w', newline='') as csvfile:
        fieldnames = ['id', 'image_id', 'category_id', 'bbox', 'score']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        txt_file = os.listdir(image_file)
        # 遍历文件夹中的所有txt文件
        id_num=0
        for txt in txt_file:
            txt =txt.split('.')[0]   # 假设文件名是1.txt, 2.txt,...,500.txt
            txt_path = os.path.join(txt_folder, txt+'.txt')
            image_id=txt.split('.')[0]
            if not os.path.exists(txt_path):
                print(f"Warning: File {txt} not found, skipping...")
                category, xmin, ymin, xmax, ymax, score = 0,0,0,0,0,0
                writer.writerow({
                    'id': id_num,
                    'image_id': image_id,
                    'category_id': category,
                    'bbox': [xmin,ymin,xmax,ymax],
                    'score': score
                })
                id_num += 1
                continue

            else:
                # 读取txt文件内容
                with open(txt_path, 'r') as f:
                    lines = f.readlines()

                # 初始化存储列表
                categories = []
                bboxes = []
                scores = []

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # 解析每行数据
                    parts = line.split()
                    if len(parts) != 6:
                        print(f"Warning: Invalid line format in {txt}: {line}")
                        continue

                    try:
                        # codetr
                        category, score, xmin, ymin, xmax, ymax = map(float, parts)
                    # yolo
                    #     category, xmin, ymin, xmax, ymax, score = map(float, parts)
                    except ValueError:

                        print(f"Warning: Invalid data in {txt}: {line}")
                        continue

                # 计算bbox格式 [x, y, width, height]
                #     xmin = max(0, xmin)
                #     xmax = min(640, xmax)
                #     ymin = max(0, ymin)
                #     ymax = min(640, ymax)
                    # codetr
                    x = (xmin+xmax)/2
                    y = (ymin+ymax)/2
                    width = xmax - xmin
                    height = ymax - ymin
                    # yolo
                    # x=xmin*640
                    # y=ymin*512
                    # width=xmax*640
                    # height = ymax*512


                    # 添加到相应列表
                    categories.append(str(int(category)))
                    bboxes.append(f"[{x:.2f}, {y:.2f}, {width:.2f}, {height:.2f}]")
                    scores.append(str(score))

            # 写入CSV行
                writer.writerow({
                    'id': id_num,
                    'image_id': image_id,
                    'category_id': ','.join(categories),
                    'bbox': ', '.join(bboxes),
                    'score': ','.join(scores)
                })
                id_num+=1

    print(f"CSV file created successfully at {output_csv}")


# 使用示例
if __name__ == "__main__":
    image_file = '/home/shen5/zhb/ODinMJ/test/rgb'
    txt_folder = '/home/shen5/zhb/ODinMJ/test/result/idat'
    output_csv = '/home/shen5/zhb/ODinMJ/test/codetr_idat.csv'
    process_txt_to_csv(image_file,txt_folder, output_csv)
