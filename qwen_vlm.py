import time
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from datasets import Dataset
import torch
import json
import os
from PIL import Image

# Load dataset
def resize_image(img_pil, max_size=640):
    """調整圖片大小"""
    width, height = img_pil.size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        return img_pil.resize((new_width, new_height), Image.LANCZOS)
    return img_pil


def get_resized_bbox(bbox, original_size, resized_size):
    """將歸一化bbox轉回原始圖片尺寸的bbox"""
    orig_w, orig_h = original_size
    new_w, new_h = resized_size

    x1_norm, y1_norm, x2_norm, y2_norm = bbox

    # Convert normalized coordinates to pixel coordinates in resized image
    x1_resized = x1_norm * new_w
    y1_resized = y1_norm * new_h
    x2_resized = x2_norm * new_w
    y2_resized = y2_norm * new_h

    # Calculate scaling factors
    scale_x = orig_w / new_w
    scale_y = orig_h / new_h

    # Convert to original image coordinates
    x1_orig = x1_resized * scale_x
    y1_orig = y1_resized * scale_y
    x2_orig = x2_resized * scale_x
    y2_orig = y2_resized * scale_y

    return [x1_orig, y1_orig, x2_orig, y2_orig]

# Load custom training data from JSON
images_dir = './training_data/images'

# Load model
base_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"


# System prompt
SYSTEM_PROMPT = """你是專業的產品辨識專家，擅長辨識飲料包裝上的文字和品牌。

當用戶提供產品位置資訊時：
1. 嚴格按照產品編號順序回答（產品 #1, #2, #3...）
2. 每個編號只描述該座標範圍內的產品
3. 仔細觀察該位置區域內的產品
4. 放大注意該區域，仔細閱讀所有可見的文字
5. 優先辨識最大、最明顯的品牌文字
6. 辨識產品名稱、口味、容量等資訊
7. 描述包裝的顏色和主要視覺特徵

重要原則：
- 產品編號與位置座標必須一一對應，絕對不可混淆
- 如果相鄰產品相似，請特別仔細區分座標範圍
- 必須仔細閱讀包裝上的所有文字，不要只看顏色就推測
- 如果某個區域的文字模糊無法辨識，請明確說明「文字模糊，無法辨識」
- 不要混淆不同位置的產品
- 品牌名稱通常是包裝上最大、最顯眼的文字
- 即使包裝相似，也要區分不同的產品和口味"""


class QwenVLM():
    def __init__(self):
        # Load model
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(base_model_id, use_fast=True, padding_side="left")

        print("Loading base model...")
        self.base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.base_model.eval()
        print("Base model loaded!")

    def generate(self, pil_image, question = "Count all products in the image."):
        pil_image = resize_image(pil_image)
        

        # Create conversation
        message = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": question},
                ],
            },
        ]

        # Apply chat template
        prompt = self.processor.apply_chat_template(message, add_generation_prompt=True, tokenize=False)

        # Generate with BASE MODEL only
        print("\nGenerating response with BASE MODEL...")
        inputs = self.processor(
            text=[prompt],
            images=[pil_image],
            return_tensors="pt",
            padding=True,
        ).to(self.base_model.device)

        with torch.no_grad():
            output_ids = self.base_model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
                # temperature=0.7,
                # top_p=0.95,
            )

        # Decode base model response
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(response)
        return response

if __name__ == "__main__":
    qwenVLM = QwenVLM()

    image_list = ["2.jpg", "3.jpg", "4.jpg", "11.jpg", "15.jpg", "101.jpg", "104.jpg"]

    image_list_d = ["2.jpg", "3.jpg", "4.jpg", "11.jpg", "15.jpg"]
    questions = [
        "圖中有幾包沙拉", 
        "圖中有幾盒紅色的餅乾盒",
        "圖中有幾個麵包的標籤是紅色的",
        "圖中有幾罐紅色的醬",
        "圖中有幾個罐頭的蓋子是米色的",
    ]

    image_list_2 =["101.jpg"]

    sum_time_elapsed_1 = 0
    sum_time_elapsed_2 = 0
    for img_name, question in zip(image_list_d, questions):

        image_path = os.path.join(images_dir, img_name)

        # calaulate time elapsed
        print('=' * 20)
        start_time_1 = time.time()
        qwenVLM.generate(image_path, question=question)
        end_time_1 = time.time()
        time_elapsed_1 = end_time_1 - start_time_1
        print(f"Time elapsed: {time_elapsed_1} seconds")
        sum_time_elapsed_1 += time_elapsed_1
        print('=' * 20)

        # print('+' * 20)
        # start_time_2 = time.time()
        # ministralVLM.generate(image_path,system_prompt_mode=2)
        # end_time_2 = time.time()
        # time_elapsed_2 = end_time_2 - start_time_2
        # print(f"Time elapsed: {time_elapsed_2} seconds")
        # sum_time_elapsed_2 += time_elapsed_2
        # print('+' * 20)

    print(f"Average Time elapsed (No Reasoning): {sum_time_elapsed_1/len(image_list)} seconds")
    print(f"Average Time elapsed (With Reasoning): {sum_time_elapsed_2/len(image_list)} seconds")

