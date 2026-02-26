from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pathlib import Path
import uvicorn
import json
import time
import os
import sys
import base64
import io

# Path setup so we can import from the network package
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
import torchvision.transforms as transforms
from PIL import Image
from network.main.network import Network

app = FastAPI()

# Data storage path
DATA_DIR = ROOT_DIR / "data" / "temp"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Session stats
stats = {
    "total": 0,
    "correct": 0
}

# Emoji classes — must match the order of training-data folders (0→🙂, 1→☹️, ...)
EMOJIS = ["🙂", "☹️", "❤️", "😭", "🤓"]

# Load trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Network(device)
model_path = ROOT_DIR / "network" / "main" / "model_98.6.pth"
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()
print(f"Model loaded from {model_path} on {device}")

# Preprocessing pipeline — mirrors training transforms, without augmentation
inference_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

def predict(image_base64: str) -> dict:
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Save last received image for debugging
    image.save(ROOT_DIR / "data" / "debug_last_received.png")
    tensor = inference_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
    probs = {emoji: round(probabilities[i].item(), 4) for i, emoji in enumerate(EMOJIS)}
    return dict(sorted(probs.items(), key=lambda item: item[1], reverse=True))


# Serve the drawing page at the root URL
@app.get("/")
async def get_index():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

@app.websocket("/ws/draw")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("iPad connected.")

    # Send initial stats
    await websocket.send_json({"type": "stats", "data": stats})

    try:
        while True:
            message = await websocket.receive_json()

            # Handle feedback
            if message.get("type") == "feedback":
                stats["total"] += 1
                if message.get("correct"):
                    stats["correct"] += 1

                label = message.get("label")
                vectors = message.get("vectors")
                if label and vectors:
                    label_dir = DATA_DIR / label
                    label_dir.mkdir(parents=True, exist_ok=True)
                    filename = f"{int(time.time()*1000)}.json"
                    file_path = label_dir / filename
                    with open(file_path, "w") as f:
                        json.dump(vectors, f)
                    print(f"Saved {label} to {file_path}")

                await websocket.send_json({"type": "stats", "data": stats})
                continue

            # Handle canvas image for prediction
            if message.get("type") == "image":
                sorted_probs = predict(message["data"])
                top_guess = list(sorted_probs.keys())[0]
                await websocket.send_json({
                    "type": "guess",
                    "guess": top_guess,
                    "probabilities": sorted_probs
                })
                continue

    except WebSocketDisconnect:
        print("iPad disconnected.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
