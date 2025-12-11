import time
from transformers import AutoModelForCausalLM, AutoProcessor
import torch
import os
from PIL import Image
from utils import resize_image


# Load custom training data from JSON
images_dir = '../vlm_20251030/training_data/images'

# Load model
base_model_id = "microsoft/Phi-3.5-vision-instruct"


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


# System prompt no YOLO
SYSTEM_PROMPT_NO_YOLO = """
    "你是一個專業的商品計數助手，負責根據圖片進行商品辨識與數量統計。"

    "你必須嚴格遵守以下規則，不可自行解讀或發揮，所有規則具有強制性："

    "【全域輸出規則 (最優先，適用所有模式)】"
    "2. 所有最終答案**必須**放在answer標籤之中 answer標籤為<answer></answer>"
    "3. 除了 think標籤 與 answer標籤 之外，不得輸出任何其他文字。"
    "4. 不可讓 think標籤 與 answer標籤 的內容互相矛盾。"
    "5. 一律只計算「最前排、完整可見」的商品。"
    "6. 一律使用『繁體中文』回答，商品品牌名稱保持原文，不可翻譯。"

    "【模式判斷 (第二優先）】"
    "7. 若使用者輸入『請進行商品盤點』，進入【全商品盤點模式】。"
    "8. 若使用者輸入非『請進行商品盤點』文字，進入【指定商品統計模式】。"

    "【全商品盤點模式】"
    "9. 列出圖片中『所有可辨識的商品』與『對應顏色』與『數量』。"
    "10. answer標籤之中的格式必須為："
    "   商品名稱(顏色):數量,商品名稱(顏色):數量"
    "11. 範例：ALISA橄欖罐頭(白):4,醃漬罐頭(紅):2,醃漬罐頭(黃):2"
    "12. 每一筆結果必須用半形逗號 , 隔開，不得換行，不得有多餘空格。"
    "13. 不可猜測、臆測、補全畫面中不存在的商品、顏色或數量。"

    "【指定商品統計模式】"
    "14. 僅統計使用者指定的商品名稱、顏色或外觀特徵。"
    "15. 只要圖片中的商品名稱『包含』使用者輸入的關鍵字即視為符合（英文忽略大小寫）。"
    "16. 關鍵字可以是名稱或外觀特徵，例如：紅、藍、綠、黃、白色瓶蓋、藍色包裝、紫色標籤。"
    "17. 不可將其他品牌或不包含關鍵字的商品納入計算。"
    "18. 在此模式下，answer標籤只能包含『單一阿拉伯數字』，例如：<answer>3</answer>"
    "19. 若圖片中不存在符合條件的商品，必須輸出："
    "   <think>找不到商品</think><answer>0</answer>"
    "20. 只要確認存在符合商品，answer標籤不可為 0。"
"""


SYSTEM_PROMPT_VLM = "你是專業的產品辨識專家，擅長辨識飲料包裝上的文字和品牌"

class PhiVLM():
    def __init__(self):
        # Load model
        print("Loading processor...")
        self.processor = AutoProcessor.from_pretrained(
            base_model_id,
            trust_remote_code=True,
            num_crops=4
        )

        print("Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype="auto", 
            device_map="cuda", 
            _attn_implementation='flash_attention_2',
            trust_remote_code=True, 
        )
        self.base_model.eval()
        print("Base model loaded!")

    def generate(self, pil_image, question="請進行商品盤點"):
        pil_image = resize_image(pil_image)
        pil_image.save('debug_input.jpg')

        # Create conversation with Phi-3.5 format
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<|image_1|>\n{question}"}
        ]

        # Apply chat template
        prompt = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Generate with BASE MODEL only
        print("\nGenerating response with BASE MODEL...")
        inputs = self.processor(
            prompt,
            [pil_image],
            return_tensors="pt"
        ).to(self.base_model.device)

        generation_args = {
            "max_new_tokens": 4096,
            "do_sample": False,
        }

        with torch.no_grad():
            output_ids = self.base_model.generate(
                **inputs,
                **generation_args,
                eos_token_id=self.processor.tokenizer.eos_token_id,
            )

        # Remove input tokens from output
        output_ids = output_ids[:, inputs['input_ids'].shape[1]:]

        # Decode response
        response = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        print(response)
        return response

if __name__ == "__main__":
    phiVLM = PhiVLM()

    image_list = ["2.jpg", "3.jpg", "4.jpg", "11.jpg", "15.jpg", "101.jpg", "104.jpg"]
    image_list_2 =["105.jpg"]
    image_list_d = ["2.jpg", "3.jpg", "4.jpg", "11.jpg", "15.jpg"]
    questions = [
        "圖中有幾包沙拉", 
        "圖中有幾盒紅色的餅乾盒",
        "圖中有幾個麵包的標籤是紅色的",
        "圖中有幾罐紅色的醬",
        "圖中有幾個罐頭的蓋子是米色的",
    ]

    question2 = [
        "請進行商品盤點"
    ]

    

    sum_time_elapsed_1 = 0
    sum_time_elapsed_2 = 0
    for img_name, question in zip(image_list_2, question2):

        image_path = os.path.join(images_dir, img_name)

        # Read image
        pil_image = Image.open(image_path)
        

        # calaulate time elapsed
        print('=' * 20)
        start_time_1 = time.time()
        phiVLM.generate(pil_image, question=question)
        end_time_1 = time.time()
        time_elapsed_1 = end_time_1 - start_time_1
        print(f"Time elapsed: {time_elapsed_1} seconds")
        sum_time_elapsed_1 += time_elapsed_1
        print('=' * 20)

    print(f"Average Time elapsed (No Reasoning): {sum_time_elapsed_1/len(image_list_d)} seconds")
    print(f"Average Time elapsed (With Reasoning): {sum_time_elapsed_2/len(image_list_d)} seconds")
