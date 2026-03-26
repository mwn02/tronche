from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import uvicorn
import json
import time
import os
import sys
import base64
import io
import shutil

# Path setup so we can import from the network package
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import torch
import torchvision.transforms as transforms
from PIL import Image
from network.with_pytorch.network import Network
from network.with_pytorch.main import crop_black

app = FastAPI()

# Serve icons and other static assets from the web directory
app.mount("/icons", StaticFiles(directory=Path(__file__).parent / "icons"), name="icons")

# Data storage path
DATA_DIR = ROOT_DIR / "data" / "temp"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Staging folder for collected training samples (PNG 32x32, ready to import into training dataset)
COLLECTED_DIR = ROOT_DIR / "data" / "collected"
COLLECTED_DIR.mkdir(parents=True, exist_ok=True)

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
model_path = ROOT_DIR / "network" / "saved_models" / "maybe_best.pth"
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()
print(f"Model loaded from {model_path} on {device}")

# Preprocessing pipeline — must match draw_emoji.py inference_transform
inference_transform = transforms.Compose([
    transforms.Lambda(crop_black),
    transforms.Resize((32, 32)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.7242779731750488], std=[0.3213667869567871]),
])

def predict(image_base64: str) -> tuple[dict, bytes]:
    """Run inference on a base64-encoded PNG. Returns (sorted_probs, png_32x32_bytes)."""
    image_bytes = base64.b64decode(image_base64)
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Save last received image for debugging
    image.save(ROOT_DIR / "data" / "debug_last_received.png")
    # Produce a 32x32 grayscale PNG for staging (mirrors training data format)
    img_32 = crop_black(image.convert("L")).resize((32, 32), Image.LANCZOS)
    buf = io.BytesIO()
    img_32.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    tensor = inference_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
    probs = {emoji: round(probabilities[i].item(), 4) for i, emoji in enumerate(EMOJIS)}
    return dict(sorted(probs.items(), key=lambda item: item[1], reverse=True)), png_bytes


# Serve the drawing page at the root URL
@app.get("/")
async def get_index():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))

@app.get("/download-data")
async def download_data():
    """Zips the data directory and returns it for download."""
    zip_path = ROOT_DIR / "data_archive"
    # Make sure we don't crash if data dir is empty or doesn't exist
    if not (ROOT_DIR / "data").exists():
        (ROOT_DIR / "data").mkdir(parents=True, exist_ok=True)
    shutil.make_archive(str(zip_path), 'zip', str(ROOT_DIR / "data"))
    return FileResponse(path=f"{zip_path}.zip", filename="tronche_data.zip", media_type="application/zip")

@app.websocket("/ws/draw")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("iPad connected.")

    # Cache the last processed 32x32 PNG per connection for staging on feedback
    last_image_bytes: bytes | None = None

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

                # Save the 32x32 PNG to the staging folder
                if label and last_image_bytes:
                    ts = int(time.time() * 1000)
                    collected_path = COLLECTED_DIR / label
                    collected_path.mkdir(parents=True, exist_ok=True)
                    with open(collected_path / f"{ts}.png", "wb") as f:
                        f.write(last_image_bytes)
                    print(f"Collected {label} → {collected_path / f'{ts}.png'}")

                await websocket.send_json({"type": "stats", "data": stats})
                continue

            # Handle canvas image for prediction
            if message.get("type") == "image":
                sorted_probs, last_image_bytes = predict(message["data"])
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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
