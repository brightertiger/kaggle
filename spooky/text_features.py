import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
stopwords = set(stopwords.words("english"))
train_data = pd.read_csv('../../data/spooky-author/download/train.csv')
test_data = pd.read_csv('../../data/spooky-author/download/test.csv')

def clean_string(x):
    table = str.maketrans('', '', string.punctuation)
    return x.lower().translate(table)

def count_word(x):
    return len(clean_string(x).split())

def count_word_unique(x):
    return len(set(clean_string(x).split()))

def word_lenght(x):
    words = clean_string(x).split()
    return np.mean([len(x) for x in words])

def count_punct(x):
    return np.sum([1 if y in string.punctuation else 0 for y in x])

def count_upper(x):
    return np.sum([y.title() == y for y in x.split()])

def count_stopword(x):
    return np.sum([1 if y in stopwords else 0 for y in clean_string(x).split()])

def count_stemwords(x):
    porter = PorterStemmer()
    x = clean_string(x).split()
    y = [porter.stem(y) for y in x]
    return np.sum([x[i] != y[i] for i in range(len(x))])

def count_noun(x):
    text = nltk.word_tokenize(clean_string(x))
    text = nltk.pos_tag(text)
    return np.sum([1 if 'NN' in x[1] else 0 for x in text])

def count_adj(x):
    text = nltk.word_tokenize(clean_string(x))
    text = nltk.pos_tag(text)
    return np.sum([1 if 'JJ' in x[1] else 0 for x in text])

def count_det(x):
    text = nltk.word_tokenize(clean_string(x))
    text = nltk.pos_tag(text)
    return np.sum([1 if 'DT' in x[1] else 0 for x in text])

def count_verb(x):
    text = nltk.word_tokenize(clean_string(x))
    text = nltk.pos_tag(text)
    return np.sum([1 if 'VB' in x[1] else 0 for x in text])

def count_pronoun(x):
    text = nltk.word_tokenize(clean_string(x))
    text = nltk.pos_tag(text)
    return np.sum([1 if 'PRP' in x[1] else 0 for x in text])

def count_chars(x):
    return len(x)

train_data['count_word'] = train_data.text.apply(count_word)
train_data['count_word_unique'] = train_data.text.apply(count_word_unique)
train_data['word_lenght'] = train_data.text.apply(word_lenght)
train_data['count_punct'] = train_data.text.apply(count_punct)
train_data['count_upper'] = train_data.text.apply(count_upper)
train_data['count_stemwords'] = train_data.text.apply(count_stemwords)
train_data['count_stopword'] = train_data.text.apply(count_stopword)
train_data['count_noun'] = train_data.text.apply(count_noun)
train_data['count_pronoun'] = train_data.text.apply(count_pronoun)
train_data['count_det'] = train_data.text.apply(count_det)
train_data['count_adj'] = train_data.text.apply(count_adj)
train_data['count_verb'] = train_data.text.apply(count_verb)
train_data['count_chars'] = train_data.text.apply(count_chars)

test_data['count_word'] = test_data.text.apply(count_word)
test_data['count_word_unique'] = test_data.text.apply(count_word_unique)
test_data['word_lenght'] = test_data.text.apply(word_lenght)
test_data['count_punct'] = test_data.text.apply(count_punct)
test_data['count_upper'] = test_data.text.apply(count_upper)
test_data['count_stemwords'] = test_data.text.apply(count_stemwords)
test_data['count_stopword'] = test_data.text.apply(count_stopword)
test_data['count_noun'] = test_data.text.apply(count_noun)
test_data['count_pronoun'] = test_data.text.apply(count_pronoun)
test_data['count_det'] = test_data.text.apply(count_det)
test_data['count_adj'] = test_data.text.apply(count_adj)
test_data['count_verb'] = test_data.text.apply(count_verb)
test_data['count_chars'] = test_data.text.apply(count_chars)

train_data['ratio_punct'] = train_data['count_punct'] / train_data['count_chars']
train_data['ratio_upper'] = train_data['count_upper'] / train_data['count_word']
train_data['ratio_stemwords'] = train_data['count_stemwords'] / train_data['count_word']
train_data['ratio_stopword'] = train_data['count_stopword'] / train_data['count_word']
train_data['ratio_noun'] = train_data['count_noun'] / train_data['count_word']
train_data['ratio_pronoun'] = train_data['count_pronoun'] / train_data['count_word']
train_data['ratio_det'] = train_data['count_det'] / train_data['count_word']
train_data['ratio_adj'] = train_data['count_adj'] / train_data['count_word']
train_data['ratio_verb'] = train_data['count_verb'] / train_data['count_word']

test_data['ratio_punct'] = test_data['count_punct'] / test_data['count_chars']
test_data['ratio_upper'] = test_data['count_upper'] / test_data['count_word']
test_data['ratio_stemwords'] = test_data['count_stemwords'] / test_data['count_word']
test_data['ratio_stopword'] = test_data['count_stopword'] / test_data['count_word']
test_data['ratio_noun'] = test_data['count_noun'] / test_data['count_word']
test_data['ratio_pronoun'] = test_data['count_pronoun'] / test_data['count_word']
test_data['ratio_det'] = test_data['count_det'] / test_data['count_word']
test_data['ratio_adj'] = test_data['count_adj'] / test_data['count_word']
test_data['ratio_verb'] = test_data['count_verb'] / test_data['count_word']

train_data.drop(['text','author'], axis=1, inplace=True)
test_data.drop(['text'], axis=1, inplace=True)

train_data.describe().T
test_data.describe().T

train_data.to_csv('../../data/spooky-author/data/train_text_feats.csv', index=False)
test_data.to_csv('../../data/spooky-author/data/test_text_feats.csv', index=False)