import os
import cv2
import numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image


def edge_detection(input_img):
    img = cv2.imread(input_img, cv2.IMREAD_UNCHANGED)
    bgr = img[:, :, :3]
    alpha = img[:, :, 3]

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray[alpha == 0] = 0

    blur = cv2.GaussianBlur(gray, (5,5), 0.3)
    edges = cv2.Canny(blur, 30, 100)
    
    return edges

def remove_edges(edges1, edges2):
    e1 = (edges1 > 0).astype(np.uint8)
    e2 = (edges2 > 0).astype(np.uint8)

    # Expand edges2 slightly to tolerate misalignment
    kernel = np.ones((3, 3), np.uint8)
    e2_dilated = cv2.dilate(e2, kernel, iterations=3)

    result = cv2.bitwise_and(e1, cv2.bitwise_not(e2_dilated))

    return (result * 255).astype(np.uint8)


input_path = 'datasets/hssd_eval'
output_path = f'{input_path}/sketches/'
os.makedirs(output_path, exist_ok=True)

renders_path = f'{input_path}/renders/**/*.png'
render_paths = glob(renders_path, recursive=True)
render_paths.sort()
render_dmg_paths = [p.replace('renders/', 'renders_dmg/') for p in render_paths]

for render, render_dmg in tqdm(zip(render_paths, render_dmg_paths)):
    # 1. Capture the original render's physical DPI and size
    with Image.open(render) as orig_pil:
        orig_dpi = orig_pil.info.get('dpi', (96, 96)) # fallback to 96 if none
        orig_size = orig_pil.size # (width, height)

    edges1 = edge_detection(render)
    edges2 = edge_detection(render_dmg)
    edges_new = remove_edges(edges1, edges2)
    
    # 2. Convert the OpenCV result to a PIL Image and match size
    edges_pil = Image.fromarray(edges_new)
    if edges_pil.size != orig_size:
        edges_pil = edges_pil.resize(orig_size, Image.NEAREST)

    parts = render.split('/')
    new_folder = f'{output_path}/{parts[-2]}'
    os.makedirs(new_folder, exist_ok=True)
    
    folder_file = '/'.join(parts[-2:])
    output_file = f'{output_path}/{folder_file}' 
    
    # 3. Save using Pillow while passing the original DPI metadata
    edges_pil.save(output_file, dpi=orig_dpi)
    print("Writing file: ", output_file)