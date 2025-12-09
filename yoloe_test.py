import os
from PIL import Image
from cv2 import resize
from ultralytics import YOLOE
# from ministral_vlm import MinistralVLM, resize_image, pil_to_base64_url
from qwen_vlm import QwenVLM, resize_image

def convert_normalized_bbox_to_pixel(bbox, image_size):
    """將歸一化bbox轉回圖片尺寸的bbox"""
    width, height = image_size

    x1_norm, y1_norm, x2_norm, y2_norm = bbox

    # 取到小數點後兩位
    x1_pixel = round(x1_norm * width, 2)
    y1_pixel = round(y1_norm * height, 2)
    x2_pixel = round(x2_norm * width, 2)
    y2_pixel = round(y2_norm * height, 2)

    return [x1_pixel, y1_pixel, x2_pixel, y2_pixel]

# ministral question prefix
MINISTRAL_QUESTION_PREFIX = "Analyze these detected products. Coordinates are normalized (0-1):\n\n"

# qwen question prefix
QWEN_QUESTION_PREFIX = "請分析以下偵測到的產品：\n\n"

# specify image path
images_dir = '../vlm_20251030/training_data/images'
image_path = os.path.join(images_dir, '104.jpg')

# read image size
image = Image.open(image_path)

image = resize_image(image, max_size=640)
width, height = image.size
print(f"Image size: {width}x{height}, type: {type(image)}")

# save resized image for reference
print("Saving resized input image as 'resized_input.jpg'...")
image.save('resized_input.jpg')
print("Resized image saved.")

# Initialize a YOLOE model
model = YOLOE("yoloe-11s-seg.pt")

names = ["bottle", "jar", "beverage"]
model.set_classes(names, model.get_text_pe(names))

# Run prediction. No prompts required.
# Use resized image to ensure bbox coordinates match the image sent to VLM
results = model.predict(image, conf=0.5)

# print(f"Detected {len(results[0].boxes)} objects with high confidence.")

str_bbox = f"{QWEN_QUESTION_PREFIX}\n\n"

for i, box in enumerate(results[0].boxes):
    xn1,yn1,xn2,yn2 = box.xyxyn[0].tolist()
    confidence = box.conf[0].item()
    label = model.names[int(box.cls[0].item())]
    bbox_pixel = convert_normalized_bbox_to_pixel([xn1, yn1, xn2, yn2], (width, height))

    # Convert to percentage for easier understanding
    x1_pct, y1_pct = int(xn1 * 100), int(yn1 * 100)
    x2_pct, y2_pct = int(xn2 * 100), int(yn2 * 100)

    str_bbox += f"產品 #{i+1}（{label}，信心度{confidence:.0%}）：\n"
    str_bbox += f"- 位置：從圖片左上角往右 {x1_pct}% 處、往下 {y1_pct}% 處開始\n"
    str_bbox += f"- 到圖片左上角往右 {x2_pct}% 處、往下 {y2_pct}% 處結束\n"
    str_bbox += f"- 像素座標：({bbox_pixel[0]:.0f}, {bbox_pixel[1]:.0f}) 到 ({bbox_pixel[2]:.0f}, {bbox_pixel[3]:.0f})\n\n"

# str_bbox += "\n請仔細觀察圖片，針對上述每個產品位置區域內的飲料，逐一辨識並說明：\n"
# str_bbox += "1. 品牌名稱（請仔細閱讀包裝上最明顯的文字）\n"
# str_bbox += "2. 產品名稱和口味\n"
# str_bbox += "3. 包裝主要顏色和特徵\n\n"
# str_bbox += "請務必仔細閱讀每個區域內飲料包裝上的所有可見文字。"

print("Bounding box details for VLM: ")
print(str_bbox)


vlm_model = QwenVLM()

vlm_model.generate(
    pil_image = image,
    question = str_bbox
)

# Show results
results[0].save('output.jpg')