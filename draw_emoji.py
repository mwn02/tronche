from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

# Tkinter is part of the Python standard library on Windows
import tkinter as tk
from tkinter import ttk

# Allow running from repo root without installing as a package
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from network.with_pytorch.network import Network  # noqa: E402


EMOJIS = ["🙂", "☹️", "❤️", "😭", "🤓"]

inference_transform = transforms.Compose(
		[
				transforms.Resize((32, 32)),
				transforms.ToTensor(),
				transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
				transforms.Grayscale(num_output_channels=1),
		]
)


def _load_state_dict(model_path: Path, device: str) -> dict:
		try:
				return torch.load(model_path, map_location=device, weights_only=True)
		except TypeError:
				# Older torch versions don't support weights_only
				return torch.load(model_path, map_location=device)


@torch.no_grad()
def predict_pil(image: Image.Image, model: Network, device: str) -> dict[str, float]:
		image = image.convert("RGB")
		tensor = inference_transform(image).unsqueeze(0).to(device)
		logits = model(tensor)
		probabilities = torch.softmax(logits, dim=1)[0]
		probs = {emoji: float(probabilities[i].item()) for i, emoji in enumerate(EMOJIS)}
		return dict(sorted(probs.items(), key=lambda item: item[1], reverse=True))


class EmojiDrawerApp:
		def __init__(self, model_path: Path, model_name: str):
				self.device = "cuda" if torch.cuda.is_available() else "cpu"
				self.model = Network(self.device)
				self.model.load_state_dict(_load_state_dict(model_path, self.device))
				self.model.eval()

				self.root = tk.Tk()
				self.root.title(f"Tronche — Draw Emoji ({model_name})")

				self.canvas_size = 420
				self.brush_width = 18

				self._build_ui()
				self._reset_image()

				self._last_x: int | None = None
				self._last_y: int | None = None

		def _build_ui(self) -> None:
				container = ttk.Frame(self.root, padding=12)
				container.grid(row=0, column=0, sticky="nsew")

				self.root.columnconfigure(0, weight=1)
				self.root.rowconfigure(0, weight=1)
				container.columnconfigure(0, weight=1)
				container.rowconfigure(0, weight=1)

				self.canvas = tk.Canvas(
						container,
						width=self.canvas_size,
						height=self.canvas_size,
						bg="white",
						highlightthickness=1,
						highlightbackground="#ccc",
				)
				self.canvas.grid(row=0, column=0, sticky="nsew")

				sidebar = ttk.Frame(container, padding=(12, 0, 0, 0))
				sidebar.grid(row=0, column=1, sticky="ns")

				ttk.Label(sidebar, text="Draw an emoji, then Predict.").grid(
						row=0, column=0, sticky="w"
				)

				btn_row = ttk.Frame(sidebar)
				btn_row.grid(row=1, column=0, sticky="ew", pady=(10, 0))
				btn_row.columnconfigure(0, weight=1)
				btn_row.columnconfigure(1, weight=1)
				btn_row.columnconfigure(2, weight=1)

				ttk.Button(btn_row, text="Predict", command=self.on_predict).grid(
						row=0, column=0, sticky="ew"
				)
				ttk.Button(btn_row, text="Clear", command=self.on_clear).grid(
						row=0, column=1, sticky="ew", padx=6
				)
				ttk.Button(btn_row, text="Save PNG", command=self.on_save).grid(
						row=0, column=2, sticky="ew"
				)

				self.guess_var = tk.StringVar(value="—")
				ttk.Label(sidebar, text="Top guess:").grid(row=2, column=0, sticky="w", pady=(12, 0))
				ttk.Label(sidebar, textvariable=self.guess_var, font=("Segoe UI Emoji", 36)).grid(
						row=3, column=0, sticky="w"
				)

				self.probs_text = tk.Text(sidebar, width=26, height=8, wrap="none")
				self.probs_text.grid(row=4, column=0, sticky="ew", pady=(8, 0))
				self.probs_text.configure(state="disabled")

				ttk.Label(
						sidebar, text=f"Device: {self.device}    Input: 32×32 grayscale"
				).grid(row=5, column=0, sticky="w", pady=(10, 0))

				ttk.Label(
						sidebar,
						text="Tip: keep strokes centered.\nClasses: 🙂 ☹️ ❤️ 😭 🤓",
						foreground="#555",
						justify="left",
				).grid(row=6, column=0, sticky="w", pady=(10, 0))

				self.canvas.bind("<ButtonPress-1>", self._on_down)
				self.canvas.bind("<B1-Motion>", self._on_move)
				self.canvas.bind("<ButtonRelease-1>", self._on_up)

		def _reset_image(self) -> None:
				self.image = Image.new("RGB", (self.canvas_size, self.canvas_size), "white")
				self.draw = ImageDraw.Draw(self.image)

		def _on_down(self, event) -> None:
				self._last_x, self._last_y = event.x, event.y

		def _on_move(self, event) -> None:
				if self._last_x is None or self._last_y is None:
						self._last_x, self._last_y = event.x, event.y
						return

				x0, y0 = self._last_x, self._last_y
				x1, y1 = event.x, event.y

				self.canvas.create_line(
						x0,
						y0,
						x1,
						y1,
						width=self.brush_width,
						fill="black",
						capstyle=tk.ROUND,
						smooth=True,
						splinesteps=36,
				)
				self.draw.line((x0, y0, x1, y1), fill="black", width=self.brush_width)

				self._last_x, self._last_y = x1, y1
				self.on_predict()

		def _on_up(self, _event) -> None:
				self._last_x, self._last_y = None, None

		def on_clear(self) -> None:
				self.canvas.delete("all")
				self._reset_image()
				self.guess_var.set("—")
				self._set_probs_text("")

		def on_save(self) -> None:
				out_dir = ROOT_DIR / "data" / "drawings"
				out_dir.mkdir(parents=True, exist_ok=True)
				out_path = out_dir / f"drawing_{int(time.time() * 1000)}.png"
				self.image.save(out_path)
				self._set_probs_text(self._get_probs_text() + f"\nSaved: {out_path}\n")

		def on_predict(self) -> None:
				probs = predict_pil(self.image, self.model, self.device)
				top_emoji = next(iter(probs.keys()))
				self.guess_var.set(top_emoji)

				lines = []
				for emoji, p in probs.items():
						lines.append(f"{emoji}  {p*100:6.2f}%")
				self._set_probs_text("\n".join(lines))

		def _set_probs_text(self, text: str) -> None:
				self.probs_text.configure(state="normal")
				self.probs_text.delete("1.0", tk.END)
				self.probs_text.insert("1.0", text)
				self.probs_text.configure(state="disabled")

		def _get_probs_text(self) -> str:
				return self.probs_text.get("1.0", tk.END).strip()

		def run(self) -> None:
				self.root.mainloop()


def main(model_name) -> int:
		parser = argparse.ArgumentParser(description=f"Draw an emoji and test {model_name}.")
		parser.add_argument(
				"--model",
				default=str(ROOT_DIR / "network" / "saved_models" / model_name),
				help=f"Path to .pth weights (default: network/saved_models/{model_name})",
		)
		parser.add_argument(
				"--smoke",
				action="store_true",
				help="Non-GUI smoke test: load model and run a blank image.",
		)
		args = parser.parse_args()

		model_path = Path(args.model)
		if not model_path.exists():
				print(f"Model not found: {model_path}")
				return 2

		if args.smoke:
				device = "cuda" if torch.cuda.is_available() else "cpu"
				model = Network(device)
				model.load_state_dict(_load_state_dict(model_path, device))
				model.eval()
				blank = Image.new("RGB", (420, 420), "white")
				probs = predict_pil(blank, model, device)
				top = next(iter(probs.keys()))
				print(f"Loaded {model_path} on {device}. Top guess on blank: {top}")
				print(probs)
				return 0

		app = EmojiDrawerApp(model_path, model_name)
		app.run()
		return 0


if __name__ == "__main__":
		raise SystemExit(main("model.pth"))
