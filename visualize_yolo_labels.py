import os
import cv2
from pathlib import Path


def parse_yolo_label(label_path):
    """
    解析 YOLO 格式的标签文件
    格式：class_id center_x center_y width height (归一化坐标 0-1)
    """
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                bboxes.append({
                    'class_id': class_id,
                    'center_x': center_x,
                    'center_y': center_y,
                    'width': width,
                    'height': height
                })
    return bboxes


def convert_yolo_to_pixel(bbox, img_width, img_height):
    """
    将 YOLO 归一化坐标转换为像素坐标
    """
    center_x = bbox['center_x'] * img_width
    center_y = bbox['center_y'] * img_height
    width = bbox['width'] * img_width
    height = bbox['height'] * img_height

    # 计算左上角和右下角坐标
    x1 = int(center_x - width / 2)
    y1 = int(center_y - height / 2)
    x2 = int(center_x + width / 2)
    y2 = int(center_y + height / 2)

    return x1, y1, x2, y2


def draw_bboxes_on_image(image_path, label_path, class_names, output_path):
    """
    在图像上绘制 bounding boxes 并保存
    """
    # 读取图像
    img = cv2.imread(image_path)
    if img is None:
        print(f"无法读取图像: {image_path}")
        return False

    img_height, img_width = img.shape[:2]

    # 解析标签
    bboxes = parse_yolo_label(label_path)

    # 绘制每个 bounding box
    for bbox in bboxes:
        x1, y1, x2, y2 = convert_yolo_to_pixel(bbox, img_width, img_height)

        # 绘制矩形框
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 添加标签文字
        class_id = bbox['class_id']
        class_name = class_names.get(class_id, f"Class {class_id}")
        label_text = f"{class_name}"

        # 文字背景
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(img, (x1, y1 - text_height - 10),
                     (x1 + text_width, y1), (0, 255, 0), -1)

        # 绘制文字
        cv2.putText(img, label_text, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # 保存图像
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"已保存: {output_path}")
    return True


def process_dataset(base_path, output_base_path, class_names):
    """
    处理整个数据集（train 和 val）
    """
    splits = ['train', 'val']
    total_processed = 0

    for split in splits:
        images_dir = os.path.join(base_path, 'images', split)
        labels_dir = os.path.join(base_path, 'labels', split)
        output_dir = os.path.join(output_base_path, split)

        if not os.path.exists(images_dir):
            print(f"图像目录不存在: {images_dir}")
            continue

        # 获取所有图像文件
        image_files = [f for f in os.listdir(images_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        print(f"\n处理 {split} 数据集，共 {len(image_files)} 张图像...")

        for img_file in image_files:
            # 构建路径
            img_path = os.path.join(images_dir, img_file)

            # 获取对应的标签文件（通常是相同的文件名但扩展名为 .txt）
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)

            # 输出路径
            output_path = os.path.join(output_dir, img_file)

            # 绘制 bounding boxes
            if draw_bboxes_on_image(img_path, label_path, class_names, output_path):
                total_processed += 1

    print(f"\n总共处理了 {total_processed} 张图像")


if __name__ == "__main__":
    # 数据集基础路径
    base_path = "/home/itrib30156/git_projects/yolo_series/train_example/custom"

    # 输出路径（保存到 results 文件夹）
    output_base_path = "/home/itrib30156/git_projects/yolo_series/train_example/custom/results"

    # 类别名称（从 custom.yaml 中获取）
    class_names = {
        0: "bottle"
    }

    # 处理数据集
    process_dataset(base_path, output_base_path, class_names)
