from math import floor
import os.path
import numpy as np

from PIL import Image, ImageDraw
init_x = [46]*7
init_y = [42, 46, 46, 46, 46, 46, 46]
cell_width = 36.3
cell_height = [43.2, 40.6, 40.6, 40.6, 40.6, 40.6, 40.6]
margin_x = 2
margin_y_top = 4
margin_y_bottom = 6
n_cells_x = 20
n_cells_y = [21, 22, 22, 22, 22, 22, 22]

def cut(img_path, base_save_path, init_x, init_y, cell_width, cell_height, n_cells_x, n_cells_y, page_number, remove_white_squares=True):
	with Image.open(img_path) as img:
		coords = get_coordinates(init_x, init_y, cell_width, cell_height, n_cells_x, n_cells_y)
		for i, coord in enumerate(coords):
			x0, x1, y0, y1 = coord[0][0], coord[1][0], coord[0][1], coord[1][1]
			img_crop = img.crop((x0, y0, x1, y1))
			
			if remove_white_squares:
				img_red_channel = np.array(img_crop.getchannel(0))
				mean = np.mean(img_red_channel)
				if mean > 245: continue # ignore empty cells

			img_resize = img_crop.resize((32, 32), )

			row_number = floor(i / n_cells_x)
			column_number = i % n_cells_x

			save_path = base_save_path + f"{page_number} ({row_number}, {column_number}).png"
			output_dir = os.path.dirname(save_path)
			if output_dir and not os.path.exists(output_dir):
				os.makedirs(output_dir)

			img_resize.save(save_path)

def get_coordinates(init_x, init_y, cell_width, cell_height, n_cells_x, n_cells_y):
	coords = []
	for j in range(n_cells_y):
		for i in range(n_cells_x):
			x, y = init_x + i * cell_width, init_y + j * cell_height
			x0, x1, y0, y1 = x + margin_x, x + cell_width - margin_x, y + margin_y_top, y + cell_height - margin_y_bottom
			coords.append([(x0, y0), (x1, y1)])
	return coords

def place_points(dataset_path, init_x, init_y, cell_width, cell_height, n_cells_x, n_cells_y):
	with Image.open(dataset_path) as img:
		draw = ImageDraw.Draw(img)
		coords = get_coordinates(init_x, init_y, cell_width, cell_height, n_cells_x, n_cells_y)
		for c in coords:
			draw.rectangle(c, outline="green", width=1)
		img.show()

for i in range(7):
	dataset_path = f"dataset/dataset-data/raw-data/page-{i+1}.png"
	img_base_save_path = f"dataset/dataset-data/generated-images/generated_imgs_p{i+1}/"
	cut(dataset_path, img_base_save_path, init_x[i], init_y[i], cell_width, cell_height[i], n_cells_x, n_cells_y[i], i+1)
	# place_points(dataset_path, init_x[i], init_y[i], cell_width, cell_height[i], n_cells_x, n_cells_y[i])