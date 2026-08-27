


import io

import torch
import torch.nn as nn

from PIL import Image

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from torchvision import transforms, models


# ==========================================
# APP + DEVICE
# ==========================================

app = FastAPI()

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================
# MODEL CLASSES
# ==========================================

CULTURE_CLASSES = [
    "East Asian",
    "Egyptian",
    "Greek",
    "Roman"
]

culture2idx = {
    culture: index
    for index, culture in enumerate(CULTURE_CLASSES)
}

idx2culture = {
    index: culture
    for culture, index in culture2idx.items()
}

print("Classes:", culture2idx)


# ==========================================
# MODEL DEFINITION
# ==========================================

class PotteryNet(nn.Module):

    def __init__(self, n_cultures):
        super().__init__()

        backbone = models.resnet18(
            weights=None
        )

        self.features = nn.Sequential(
            *list(backbone.children())[:-1]
        )

        self.culture_head = nn.Linear(
            512,
            n_cultures
        )

    def forward(self, x):

        x = self.features(x).flatten(1)

        return self.culture_head(x)


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = PotteryNet(
    n_cultures=len(CULTURE_CLASSES)
).to(device)

model.load_state_dict(
    torch.load(
        "best_pottery_model_cpu.pt",
        map_location=device
    )
)

model.eval()

print("Model loaded successfully.")


# ==========================================
# IMAGE PREPROCESSING
# ==========================================

inference_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ==========================================
# PREDICTION
# ==========================================

def predict_culture(image):

    image = image.convert("RGB")

    tensor = inference_transform(image)

    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():

        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    results = {}

    for index, probability in enumerate(probabilities):

        culture = idx2culture[index]

        results[culture] = float(
            probability.cpu().item()
        )

    results = dict(
        sorted(
            results.items(),
            key=lambda item: item[1],
            reverse=True
        )
    )

    return results


# ==========================================
# STATIC FILES
# ==========================================

app.mount(
    "/assets",
    StaticFiles(directory="assets"),
    name="assets"
)


# ==========================================
# HOMEPAGE
# ==========================================

@app.get("/")
def homepage():

    return FileResponse(
        "index.html"
    )


# ==========================================
# PREDICTION API
# ==========================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    image = Image.open(
        io.BytesIO(contents)
    )

    predictions = predict_culture(image)

    return {
        "predictions": predictions
    }