from fastapi import FastAPI, File, UploadFile
from app.model import predict_image
from app.utils import prepare_image

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Cat Detector API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = prepare_image(contents)
    predictions = predict_image(img)

    # Перевіримо, чи є серед top-3 класів щось про кота
    is_cat = any("cat" in label.lower() for (_, label, _) in predictions)

    return {
        "is_cat": is_cat,
        "top_predictions": [
            {"label": label, "confidence": float(score)} for (_, label, score) in predictions
        ]
    }


