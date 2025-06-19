from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import os
import torch
import torchvision.transforms as transforms
from app.model import CaptchaCNN, decode_output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CaptchaCNN().to(device)
model.load_state_dict(torch.load(os.path.join("app", "captcha_model.pth"), map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((40, 100)),
    transforms.ToTensor()
])

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Captcha solver is up and running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            prediction = decode_output(output)[0]

        return JSONResponse(content={"prediction": prediction})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
