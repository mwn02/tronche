#code tomas p2-7
import os.path

from PIL import Image, ImageDraw
initiales_x = [         ]
initiales_y = [         ]
cell_width = [          ]
cell_height = []
margin_x = []
margin_y_top = []
margin_y_bottom = []
n_cells_x = []
n_cells_y = []
for n in range(2,8): 
    dataset_path = f"dataset/page-{n}.png"
    img_base_save_path = f"dataset/generate_imgs_p{n}/"
    def cut():
        with Image.open(dataset_path) as img: 
            for j in range(n_cells_y[n-2]):
                for i in range(n_cells_x[n-2]):
                    x, y = int(initiales_x[n-2] + i*cell_width[n-2]), int(initiales_y[n-2] + j*cell_height[n-2])
                    x_0, x_1, y_0, y_1 = int(x+margin_x[n-2]), int(x+cell_width[n-2]-margin_x[n-2]), int(y + margin_y_top[n-2]), int(y + cell_height[n-2] - margin_y_bottom[n-2])

                    img_crop = img.crop((x_0, y_0, x_1, y_1))
                    img_resize = img_crop.resize((32, 32), )

                    save_path = img_base_save_path + f"{i}-{j}.png"

                    output_dir = os.path.dirname(save_path)
                    if output_dir and not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    img_resize.save(save_path)





