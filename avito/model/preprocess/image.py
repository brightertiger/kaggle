import pandas as pd
from PIL import Image
from multiprocessing import Pool

def resize(image):
    path = '../data/download/data/competition_files/train_jpg/' + image + '.jpg'
    image = Image.open(path)
    image = image.resize((150,150), Image.ANTIALIAS)
    image.save(path)
    return None

data = pd.read_csv('../data/data/image/train_data.csv', usecols=['image'])
data = data['image'].tolist()

pool = Pool(8)
pool.map(resize, data)
pool.close()

def resize(image):
    try:
        path = '../data/download/data/competition_files/test_jpg/' + image + '.jpg'
        image = Image.open(path)
        image = image.resize((150,150), Image.ANTIALIAS)
        image.save(path)
    except:
        print('error:', image)
    return None

data = pd.read_csv('../data/data/image/test_data.csv', usecols=['image'])
data = data['image'].tolist()

pool = Pool(8)
pool.map(resize, data)
pool.close()