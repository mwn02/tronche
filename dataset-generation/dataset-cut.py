# this script cuts the database's pdf into single emoji images
import os.path

from PIL import Image, ImageDraw

dataset_path = "../dataset/page-1.png"

init_x = 46
init_y = 42
cell_width = 36.3
cell_height = 43.2
margin_x = 2
margin_y_top = 4
margin_y_bottom = 6
n_cells_x = 20
n_cells_y = 21

img_base_save_path = "generate_imgs/"

def cut():
    with Image.open(dataset_path) as img:
        for j in range(n_cells_y):
            for i in range(n_cells_x):
                x, y = int(init_x + i*cell_width), int(init_y + j*cell_height)
                x_0, x_1, y_0, y_1 = int(x+margin_x), int(x+cell_width-margin_x), int(y + margin_y_top), int(y + cell_height - margin_y_bottom)

                img_crop = img.crop((x_0, y_0, x_1, y_1))
                img_crop.resize((32, 32), )

                save_path = img_base_save_path + f"{i}-{j}.png"

                output_dir = os.path.dirname(save_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                img_crop.save(save_path)

def get_coordinates():
    coords = []
    for j in range(n_cells_y):
        for i in range(n_cells_x):
            x, y = init_x + i * cell_width, init_y + j * cell_height
            x_0, x_1, y_0, y_1 = x + margin_x, x + cell_width - margin_x, y + margin_y_top, y + cell_height - margin_y_bottom
            coords.append([(x_0, y_0), (x_1, y_1)])
    return coords

def place_points():
    with Image.open(dataset_path) as img:
        draw = ImageDraw.Draw(img)
        coords = get_coordinates()
        for c in coords:
            draw.rectangle(c, outline="green", width=1)

        img.show()

cut()

