import pandas as pd
import dask.dataframe as dd
import numpy as np
import re
import string
from dask.multiprocessing import get
from collections import Counter
from gensim.models import FastText

def clean_text(text):
    text = text.lower()
    ipaddress = re.findall( r'[0-9]+(?:\.[0-9]+){3}', text)
    for ip in ipaddress:
        text = text.replace(ip,' ')
    text = text.replace('\n', ' ')
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', text)
    text = re.sub('[^A-Za-z0-9]+', ' ', text)
    text = re.sub('\s+', ' ', text)
    return text.strip().lower()

train_data = pd.read_csv('../data/download/train.csv', usecols=['comment_text'])
test_data = pd.read_csv('../data/download/test.csv', usecols=['comment_text'])
internal = train_data.append(test_data)
internal['comment_text'] = internal['comment_text'].fillna('nan')
internal = dd.from_pandas(internal, npartitions=10)
internal = internal.map_partitions(lambda df: df.apply((lambda row: clean_text(*row)),axis=1))
internal = internal.compute(get=get)
internal = pd.DataFrame(internal, columns=['comment_text'])
del train_data, test_data
print('internal data:', internal.shape)

data = internal.copy()
data['comment_text'] = data['comment_text'].fillna('nan')
data['comment_text'] = data['comment_text'].apply(lambda x : x.split())
vocab = data['comment_text'].values
vocab = [y for x in vocab for y in x if not re.match('.*\d+', y)] 
vocab = Counter(vocab)
vocab = dict(vocab)

model = FastText.load_fasttext_format('../data/data/fasttext/wiki.en.bin')

file = open('../data/data/fasttext/fasttext.txt','w')
for word in vocab.keys():
    file.write(word)
    file.write(' ')
    file.write(' '.join([str(x) for x in model.wv.get_vector(word).tolist()]))
    file.write('\n')
file.close()