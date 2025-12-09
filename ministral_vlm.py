import os
import torch
import torch
from PIL import Image
from unsloth import FastVisionModel # FastLanguageModel for LLMs
from transformers import TextStreamer
import time

SYSTEM_PROMPT_FULL_INVENTORY = """You are a product recognition expert. Carefully analyze each product in the given bounding boxes.
For each bbox:
1. Read ALL text visible on the product (brand name, flavor, description)
2. Describe the packaging color and design
3. Identify the exact product name and variant
4. Be precise - do not confuse similar-looking products

The bounding box coordinates are normalized (0-1 range), where (x1, y1) is top-left and (x2, y2) is bottom-right."""




images_dir = './training_data/images'
model_id = "unsloth/Ministral-3-3B-Instruct-2512"

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

# Load custom training data from JSON



def pil_to_base64_url(img_pil):
    import base64
    from io import BytesIO

    buffered = BytesIO()
    img_pil.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"



class MinistralVLM():
    def __init__(self):
        
        self.model, self.tokenizer = FastVisionModel.from_pretrained(
            model_id,
            load_in_4bit = False, # Use 4bit to reduce memory use. False for 16bit LoRA.
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
        )

        self.model = FastVisionModel.get_peft_model(
            self.model,
            finetune_vision_layers     = True, # False if not finetuning vision layers
            finetune_language_layers   = True, # False if not finetuning language layers
            finetune_attention_modules = True, # False if not finetuning attention layers
            finetune_mlp_modules       = True, # False if not finetuning MLP layers

            r = 32,           # The larger, the higher the accuracy, but might overfit
            lora_alpha = 32,  # Recommended alpha == r at least
            lora_dropout = 0,
            bias = "none",
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
            # target_modules = "all-linear", # Optional now! Can specify a list if needed
        )

        FastVisionModel.for_inference(self.model) # Enable for inference!

    def generate(self, pil_image, question = "Count all products in the image.", system_prompt_mode=1):
        # img_pil = Image.open(image_path).convert("RGB")
        pil_image = resize_image(pil_image)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_FULL_INVENTORY},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image"},
                ],
            },
        ]

        input_text = self.tokenizer.apply_chat_template(messages, add_generation_prompt = True)
        inputs = self.tokenizer(
            pil_image,
            input_text,
            add_special_tokens = False,
            return_tensors = "pt",
        ).to("cuda")

        
        # print(123123123)
        text_streamer = TextStreamer(self.tokenizer, skip_prompt = True)
        # print(123123123)
        decoded_output = self.model.generate(**inputs, streamer = text_streamer, max_new_tokens = 1000,
                        use_cache = True, temperature = 0.1, min_p = 0.05, do_sample = False)
        
        # print(123123123)
        # print(decoded_output)

        # messages = [
        #     {"role": "system", "content": system_prompt},
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "text", "text": question},
        #             {"type": "image_url", "image_url": {"url": image_url}},
        #         ],
        #     },
        # ]

        # tokenized = self.tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True)

        # tokenized["input_ids"] = tokenized["input_ids"].to(device="cuda")
        # tokenized["pixel_values"] = tokenized["pixel_values"].to(dtype=torch.bfloat16, device="cuda")
        # image_sizes = [tokenized["pixel_values"].shape[-2:]]

        # output = self.model.generate(
        #     **tokenized,
        #     image_sizes=image_sizes,
        #     max_new_tokens=512,
        # )[0]

        # decoded_output = self.tokenizer.decode(output[len(tokenized["input_ids"][0]):])
        return decoded_output


