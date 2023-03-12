import os
from typing import Tuple

import cv2
from PIL import Image, ImageDraw
from bs4 import BeautifulSoup
from torchvision.transforms import *

org_path = "../dataset/sig-train-pos"
org_path_neg = "../dataset/sig-train-neg"
target_img_dir = "../dataset/images"
target_label_dir = "../dataset/labels"
target_display_dir = "../dataset/display"


def mkdir_with_check(dir: str):
    if not os.path.exists(dir):
        os.mkdir(dir)
        print("Directory ", dir, " Created ")
    else:
        print("Directory ", dir, " already exists")


def prepare_dir():
    mkdir_with_check(target_img_dir)
    mkdir_with_check(os.path.join(target_img_dir, "train"))
    mkdir_with_check(os.path.join(target_img_dir, "val"))

    mkdir_with_check(target_label_dir)
    mkdir_with_check(os.path.join(target_label_dir, "train"))
    mkdir_with_check(os.path.join(target_label_dir, "val"))

    mkdir_with_check(target_display_dir)


def list_files_with_extension(path, extension):
    files = []
    for file in os.listdir(path):
        if file.endswith(extension):
            files.append(file)
    return files


def prepare(img_dir: str, label_dir: str, img_name: str, xml_name: str):
    img_src = os.path.join(org_path, img_name)

    print(f'Dealing with {img_name}... ', end='')

    img: Image.Image = Image.open(img_src)
    xml_path = os.path.join(org_path, xml_name)
    boxes = parse_xml(xml_path)

    sub_images = []
    sub_boxes = []
    sub_width = img.width // 2
    sub_height = img.height // 2
    for i in range(2):
        for j in range(2):
            # 计算子图片的左上角坐标和右下角坐标
            left = j * sub_width
            upper = i * sub_height
            right = left + sub_width
            lower = upper + sub_height
            # 裁剪子图片
            sub_image = img.crop((left, upper, right, lower))
            # 存储子图片
            sub_images.append(sub_image)
            # 计算子图片中所有矩形框的新坐标
            sub_box = []
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                if xmin >= left and ymin >= upper and xmax <= right and ymax <= lower:
                    # 如果矩形框在子图片内，则计算新坐标
                    sub_box.append([xmin - left, ymin - upper, xmax - left, ymax - upper])
            # 存储子图片中所有矩形框的新坐标
            sub_boxes.append(sub_box)
    # 存储四个子图片和对应的label文件
    for i in range(4):
        sub_img_name = img_name.replace('.jpeg', f'_{i}.jpeg')
        img_tgt = os.path.join(img_dir, sub_img_name)
        # 存储子图片
        sub_images[i].save(img_tgt)
        # 存储子图片中所有矩形框的新坐标和对应的label文件
        label_path = os.path.join(label_dir, xml_name.replace('.xml', f'_{i}.txt'))
        with open(label_path, "w") as f:
            for box in sub_boxes[i]:
                xmin, ymin, xmax, ymax = box
                x_center = (xmin + xmax) / 2 / sub_images[i].width
                y_center = (ymin + ymax) / 2 / sub_images[i].height
                width = (xmax - xmin) / sub_images[i].width
                height = (ymax - ymin) / sub_images[i].height
                # 将矩形框信息写入label文件
                f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
            out_img_path = os.path.join(target_display_dir, sub_img_name)
            plot_boxes(sub_images[i], sub_boxes[i], out_img_path)
    print('Done')


def plot_boxes(img: Image.Image, boxes, out_path: str):
    draw = ImageDraw.Draw(img)
    for box in boxes:
        left, upper, right, lower = box
        # 将矩形框画到图片上
        draw.rectangle([left, upper, right, lower], outline="red", width=3)
    img.save(out_path)


def do_augment(img_path: str, out_path: str, transform):
    img = Image.open(img_path)  # 转换为RGB格式
    img_tensor = transform(img)
    transform_back = transforms.ToPILImage()
    img_pil = transform_back(img_tensor)
    img_pil.save(out_path)


def parse_xml(xml_path: str):
    with open(xml_path) as f:
        # image_width, image_height = image_shape
        data = f.read()
        soup = BeautifulSoup(data, 'lxml')
        box_tags = soup.find_all('bndbox')
        boxes = []
        for box_tag in box_tags:
            xmin = int(box_tag.xmin.string)
            ymin = int(box_tag.ymin.string)
            xmax = int(box_tag.xmax.string)
            ymax = int(box_tag.ymax.string)

            boxes.append((xmin, ymin, xmax, ymax))
        return boxes


if __name__ == '__main__':
    prepare_dir()

    xmls = list_files_with_extension(org_path, ".xml")
    imgs = list_files_with_extension(org_path, ".jpeg")

    num = len(xmls)
    assert num == len(imgs)

    val_percent = 0.2
    num_val = int(num * val_percent)
    num_train = num - num_val

    train_img_path = os.path.join(target_img_dir, 'train')
    train_label_path = os.path.join(target_label_dir, 'train')
    for i in range(num_train):
        img_name = imgs[i]
        xml_name = xmls[i]
        prepare(train_img_path, train_label_path, img_name, xml_name)

    val_img_path = os.path.join(target_img_dir, 'val')
    val_label_path = os.path.join(target_label_dir, 'val')
    for i in range(num_train, num):
        img_name = imgs[i]
        xml_name = xmls[i]
        prepare(val_img_path, val_label_path, img_name, xml_name)

    # imgs = list_files_with_extension(org_path_neg, ".jpeg")
    #
    # num = int(len(imgs) * 0.2)
    # num_val = int(num * val_percent)
    # num_train = num - num_val
    #
    # for i in range(num_train):
    #     img_name = imgs[i]
    #     img_src = os.path.join(org_path_neg, img_name)
    #     img_tgt = os.path.join(train_img_path, img_name)
    #     shutil.copy(img_src, img_tgt)
    #
    # for i in range(num_train, num):
    #     img_name = imgs[i]
    #     img_src = os.path.join(org_path_neg, img_name)
    #     img_tgt = os.path.join(val_img_path, img_name)
    #     shutil.copy(img_src, img_tgt)
