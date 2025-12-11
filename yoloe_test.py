import os

import cv2
import math
from ultralytics import YOLOE
from PIL import Image
# from ministral_vlm import MinistralVLM, resize_image, pil_to_base64_url
from phi_vlm import PhiVLM
from qwen_vlm import QwenVLM
from utils import resize_image, convert_normalized_bbox_to_pixel


# ministral question prefix
MINISTRAL_QUESTION_PREFIX = "Analyze these detected products. Coordinates are normalized (0-1):\n\n"

# qwen question prefix
QWEN_QUESTION_PREFIX = "請分析以下偵測到的產品：\n\n"

# specify image path
# images_dir = '../vlm_20251030/training_data/images'
# image_path = os.path.join(images_dir, '104.jpg')

# # read image size
# image = Image.open(image_path)

# image = resize_image(image, max_size=640)
# width, height = image.size
# print(f"Image size: {width}x{height}, type: {type(image)}")


# start webcam
cap = cv2.VideoCapture(0)
WIDTH = 640
HEIGHT = 480
cap.set(3, WIDTH)
cap.set(4, HEIGHT)


# Initialize a YOLOE model
model = YOLOE("yoloe-11s-seg.pt")

names = ["bottle", "beverage"]
model.set_classes(names, model.get_text_pe(names))

# initial VLM
# vlm_model = QwenVLM()

while True:
    success, img = cap.read()
    results = model(img, stream=True, conf=0.4)

    # coordinates
    r = list(results)[0]
    boxes = r.boxes

    str_bbox = f"{QWEN_QUESTION_PREFIX}\n\n"

    for i, box in enumerate(boxes):
        # prepare message for vlm
        xn1,yn1,xn2,yn2 = box.xyxyn[0].tolist()
        confidence = box.conf[0].item()
        label = model.names[int(box.cls[0].item())]
        bbox_pixel = convert_normalized_bbox_to_pixel([xn1, yn1, xn2, yn2], (WIDTH, HEIGHT))

        # Convert to percentage for easier understanding
        x1_pct, y1_pct = int(xn1 * 100), int(yn1 * 100)
        x2_pct, y2_pct = int(xn2 * 100), int(yn2 * 100)

        str_bbox += f"產品 #{i+1}（{label}，信心度{confidence:.0%}）：\n"
        str_bbox += f"- 位置：從圖片左上角往右 {x1_pct}% 處、往下 {y1_pct}% 處開始\n"
        str_bbox += f"- 到圖片左上角往右 {x2_pct}% 處、往下 {y2_pct}% 處結束\n"
        str_bbox += f"- 像素座標：({bbox_pixel[0]:.0f}, {bbox_pixel[1]:.0f}) 到 ({bbox_pixel[2]:.0f}, {bbox_pixel[3]:.0f})\n\n"

        # bounding box
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

        xn1,yn1,xn2,yn2 = box.xyxyn[0].tolist()
        confidence = box.conf[0].item()
        label = model.names[int(box.cls[0].item())]
        bbox_pixel = convert_normalized_bbox_to_pixel([xn1, yn1, xn2, yn2], (WIDTH, HEIGHT))

        # put box in cam
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 3)

        # confidence
        confidence = math.ceil((box.conf[0]*100))/100
        # print("Confidence --->",confidence)

        # class name
        label_id = int(box.cls[0])

        # object details
        org = [x1, y1 - 4]
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.6
        color = (255, 0, 0)
        thickness = 2


        cv2.putText(img, f'{names[label_id]} {confidence:.2f}', org, font, fontScale, color, thickness)


    # convert opencv image to PIL image, BGR to RGB
    # pil_image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # vlm_model.generate(
    #     pil_image = pil_image,
    #     question = str_bbox
    # )


    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# # Run prediction. No prompts required.
# # Use resized image to ensure bbox coordinates match the image sent to VLM
# results = model.predict(image, conf=0.5)

# # print(f"Detected {len(results[0].boxes)} objects with high confidence.")

# str_bbox = f"{QWEN_QUESTION_PREFIX}\n\n"

# for i, box in enumerate(results[0].boxes):
#     xn1,yn1,xn2,yn2 = box.xyxyn[0].tolist()
#     confidence = box.conf[0].item()
#     label = model.names[int(box.cls[0].item())]
#     bbox_pixel = convert_normalized_bbox_to_pixel([xn1, yn1, xn2, yn2], (width, height))

#     # Convert to percentage for easier understanding
#     x1_pct, y1_pct = int(xn1 * 100), int(yn1 * 100)
#     x2_pct, y2_pct = int(xn2 * 100), int(yn2 * 100)

#     str_bbox += f"產品 #{i+1}（{label}，信心度{confidence:.0%}）：\n"
#     str_bbox += f"- 位置：從圖片左上角往右 {x1_pct}% 處、往下 {y1_pct}% 處開始\n"
#     str_bbox += f"- 到圖片左上角往右 {x2_pct}% 處、往下 {y2_pct}% 處結束\n"
#     str_bbox += f"- 像素座標：({bbox_pixel[0]:.0f}, {bbox_pixel[1]:.0f}) 到 ({bbox_pixel[2]:.0f}, {bbox_pixel[3]:.0f})\n\n"

# # str_bbox += "\n請仔細觀察圖片，針對上述每個產品位置區域內的飲料，逐一辨識並說明：\n"
# # str_bbox += "1. 品牌名稱（請仔細閱讀包裝上最明顯的文字）\n"
# # str_bbox += "2. 產品名稱和口味\n"
# # str_bbox += "3. 包裝主要顏色和特徵\n\n"
# # str_bbox += "請務必仔細閱讀每個區域內飲料包裝上的所有可見文字。"

# print("Bounding box details for VLM: ")
# print(str_bbox)




# vlm_model.generate(
#     pil_image = image,
#     question = str_bbox
# )

# Show results
# results[0].show()