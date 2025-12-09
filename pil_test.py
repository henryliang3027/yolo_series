import os
from PIL import Image

from qwen_vlm import resize_image


images_dir = '../vlm_20251030/training_data/images'
image_path = os.path.join(images_dir, '1.jpg')


image = Image.open(image_path)
image = resize_image(image, max_size=640)

# image.show()

image.save('output_image.jpg')