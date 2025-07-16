from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image

def prepare_image(file):
    img = Image.open(io.BytesIO(file)).convert("RGB").resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 127.5 - 1.0  # MobileNetV2 expects [-1, 1]
    return img_array
