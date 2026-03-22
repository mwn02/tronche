# TRONCHE
## Technical Recognition of Optical Networks for Complex Handrawn Emojis

## Installation

1. Clone the repository
2. Install dependencies:
   pip install -r requirements.txt
   
# Quick test (draw in Python)

Run the desktop drawing app (loads `network/main/model_94.pth` by default):

```bash
python draw_emoji.py
```

Optional smoke test (no GUI, just verifies the model loads and runs inference once):

```bash
python draw_emoji.py --smoke
```
# Run benchmarks

All benchmarks (50 epochs)

```bash
python network/with_pytorch/benchmarks/run_all_benchmarks.py
```
# projet d'intelligence artificielle par les prophètes emojis 😎
