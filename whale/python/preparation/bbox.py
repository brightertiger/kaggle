import pandas as pd 
import numpy as np 
import glob 
import cv2
from tqdm import tqdm

def clip_image(csv_file, in_path, out_path):
    data = pd.read_csv(csv_file)
    data = data.set_index('Image')
    files = glob.glob(in_path + '/*.jpg')
    files = [x.split('/')[-1] for x in files]
    for file in tqdm(files):
        image = cv2.imread(in_path + file)
        box = data.loc[file,:]
        x0, y0, x1, y1 = box['x0'], box['y0'], box['x1'],  box['y1']
        if not (x0 >= x1 or y0 >= y1):
            image = image[y0:y1, x0:x1,:]
        cv2.imwrite(out_path + file, image)
    return None