import numpy as np
import os
from PIL import Image, ImageDraw
import fitz

doc = fitz.open("dataset/dataset-data/raw-data/Notes_260205_112327.pdf")

PAGE_NUM = 0 

page = doc[PAGE_NUM]
pix = page.get_pixmap(dpi=300)
img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
W, H = img.size

# Paramètres de ta grille
ROWS = 10
COLS = 15          # nombre de coeurs par ligne (à découper)
TOTAL_COLS = 15.8    # largeur réelle de la page en colonnes (1 col de marge à droite)

# La taille d'une cellule est basée sur la largeur totale (15.5 cols)
w_cell = W / TOTAL_COLS
h_cell = H / ROWS
margin = 10

# Dessin de la grille pour vérification
img_debug = img.copy()
draw = ImageDraw.Draw(img_debug)
for r in range(ROWS):
	for c in range(COLS):
		left = c * w_cell + margin
		top = r * h_cell + margin
		right = (c + 1) * w_cell - margin
		bottom = (r + 1) * h_cell - margin
		draw.rectangle([left, top, right, bottom], outline="green", width=1)

img_debug_path = "dataset/dataset-data/debug"
if not os.path.exists(img_debug_path):
	os.makedirs(img_debug_path)
img_debug.save(f"{img_debug_path}/debug_grid.png")
print(f"Fichier debug_grid.png créé dans {img_debug_path}. Ouvre-le pour vérifier l'alignement !")

# ==========================================
# 3. DÉCOUPAGE ET SAUVEGARDE
# ==========================================
output_dir = "dataset/dataset-data/generated-images/charles_emojis"
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

print("Découpage en cours...")
count = 0
# On boucle sur les lignes et les colonnes (on ignore la 15ème si elle est vide/tronquée)
for p in range(8):
	page = doc[PAGE_NUM+p]
	pix = page.get_pixmap(dpi=300)
	img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
	for r in range(ROWS):
		for c in range(COLS):
			left = c * w_cell + margin
			top = r * h_cell + margin
			right = (c + 1) * w_cell - margin
			bottom = (r + 1) * h_cell - margin
			
			# Crop (gauche, haut, droite, bas)
			emoji = img.crop((left, top, right, bottom))

			img_red_channel = np.array(emoji.getchannel(0))
			mean = np.mean(img_red_channel)
			# print(mean)
			if mean > 253:
				print(f"Empty cell at {p+1}:{r+1}_{c+1}")
				continue

			emoji = emoji.resize((32, 32), Image.LANCZOS)
			emoji = emoji.point(lambda x: (x/256)**2 * 256)  # Seuil pour rendre l'image binaire (noir et blanc)
			# Sauvegarde
			emoji.save(f"{output_dir}/emoji_{p+1}{r+1}_{c+1}.png")
			count += 1

print(f"Terminé ! {count} emojis sauvegardés dans le dossier '{output_dir}'.")
