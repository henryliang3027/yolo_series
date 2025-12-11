from PIL import Image
from io import BytesIO
import base64


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


def pil_to_base64_url(img_pil):
    buffered = BytesIO()
    img_pil.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

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