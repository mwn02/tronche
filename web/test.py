from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pathlib import Path
import uvicorn
import json
import random
import time
import os

app = FastAPI()

# Data storage path
DATA_DIR = Path("data/temp")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Session stats
stats = {
    "total": 0,
    "correct": 0
}

# Serve the drawing page at the root URL
@app.get("/")
async def get_index():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text())

@app.websocket("/ws/draw")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("iPad connected.")
    
    # Send initial stats
    await websocket.send_json({"type": "stats", "data": stats})
    
    try:
        while True:
            # Receive data
            message = await websocket.receive_json()
            
            # Handle Feedback
            if message.get("type") == "feedback":
                stats["total"] += 1
                if message.get("correct"):
                    stats["correct"] += 1
                
                # Save data
                label = message.get("label")
                vectors = message.get("vectors")
                if label and vectors:
                    # Create directory for the label
                    label_dir = DATA_DIR / label
                    label_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save file
                    filename = f"{int(time.time()*1000)}.json"
                    file_path = label_dir / filename
                    with open(file_path, "w") as f:
                        json.dump(vectors, f)
                    print(f"Saved {label} to {file_path}")

                # Broadcast new stats
                await websocket.send_json({"type": "stats", "data": stats})
                continue
            
            # Handle Drawing Data (assume it's coordinate data if not 'feedback')
            # This waits silently until the iPad sends a coordinate
            data = message
            
            # This is where you'll eventually feed the math to the neural network
            # print(f"Received raw input: {data}")
            
            # Mock proabilities for the "Advanced Mode"
            # In real implementation: probs = model.predict(image)
            emojis = ["🐢", "🦕", "🦎", "🐍", "🐊"]
            probs = {}
            remaining = 1.0
            
            # Generate random probabilities that sum to ~1
            for i, emoji in enumerate(emojis[:-1]):
                p = round(random.uniform(0, remaining * 0.8), 2)
                probs[emoji] = p
                remaining -= p
            probs[emojis[-1]] = round(remaining, 2)
            
            # Sort by probability
            sorted_probs = dict(sorted(probs.items(), key=lambda item: item[1], reverse=True))
            top_guess = list(sorted_probs.keys())[0]

            response = {
                "type": "guess",
                "guess": top_guess,
                "probabilities": sorted_probs
            }

            await websocket.send_json(response)
            
    except WebSocketDisconnect:
        print("iPad disconnected.")

if __name__ == "__main__":
    # Runs the server on your local network on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)