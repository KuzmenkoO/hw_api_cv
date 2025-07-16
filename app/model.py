from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input

model = MobileNetV2(weights="imagenet")

def predict_image(img_array):
    preds = model.predict(img_array)
    decoded = decode_predictions(preds, top=3)[0]
    return decoded
