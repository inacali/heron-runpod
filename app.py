# app.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import torch
from transformers import AutoImageProcessor, RTDetrV2ForObjectDetection

MODEL_ID = "docling-project/docling-layout-heron-101"

app = FastAPI(title="Docling Heron-101 Layout API")

# Load model & processor at startup
processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = RTDetrV2ForObjectDetection.from_pretrained(MODEL_ID)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

id2label = model.config.id2label


@app.get("/health")
async def health():
    return {"status": "ok", "device": str(device)}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image bytes
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Prepare inputs
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # post_process; note: target_sizes expects (height, width)
    target_sizes = [image.size[::-1]]  # (H, W)
    results = processor.post_process_object_detection(
        outputs,
        threshold=0.3,  # tweak as needed
        target_sizes=target_sizes,
    )[0]

    predictions = []
    for score, label, box in zip(
        results["scores"], results["labels"], results["boxes"]
    ):
        # box is [x_min, y_min, x_max, y_max] in pixels
        x_min, y_min, x_max, y_max = box.tolist()
        label_id = int(label)
        label_name = id2label.get(label_id, str(label_id))

        predictions.append(
            {
                "bbox": [x_min, y_min, x_max, y_max],
                "score": float(score),
                "label_id": label_id,
                "label": label_name,
            }
        )

    return JSONResponse(content={"predictions": predictions})

