import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn import metrics
import string
pd.options.mode.chained_assignment = None
stopwords = set(stopwords.words('english'))

train_data = pd.read_csv('../../data/spooky-author/download/train.csv')
test_data = pd.read_csv('../../data/spooky-author/download/test.csv')

vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,3))
train_text = list(train_data.text.values)
test_text = list(test_data.text.values)
full_text = train_text + test_text
full_text = vectorizer.fit_transform(full_text)
train_word  = vectorizer.transform(train_text)
test_word  = vectorizer.transform(test_text)
print('train:', train_word.shape)
print('test:', test_word.shape)

vectorizer = CountVectorizer(ngram_range=(1,7), analyzer='char')
train_text = list(train_data.text.values)
test_text = list(test_data.text.values)
full_text = train_text + test_text
full_text = vectorizer.fit_transform(full_text)
train_char_cnt  = vectorizer.transform(train_text)
test_char_cnt  = vectorizer.transform(test_text)
print('train:', train_char_cnt.shape)
print('test:', test_char_cnt.shape)

vectorizer =  TfidfVectorizer(ngram_range=(1,5), analyzer='char')
train_text = list(train_data.text.values)
test_text = list(test_data.text.values)
full_text = train_text + test_text
full_text = vectorizer.fit_transform(full_text)
train_char_tf  = vectorizer.transform(train_text)
test_char_tf  = vectorizer.transform(test_text)
print('train:', train_char_tf.shape)
print('test:', test_char_tf.shape)

mapper = {'EAP':0, 'HPL':1, 'MWS':2}
train_data['author'] = train_data['author'].map(lambda x : mapper[x])
dependent = np.array(train_data['author'])

def naive_bayes(train_feat, test_feat):
    
    folds = KFold(n_splits=5, random_state=2017, shuffle=True)
    pred_train = np.zeros((train_data.shape[0], 3))
    pred_test = np.zeros((test_data.shape[0], 3))
    
    for dev_index, val_index in folds.split(train_data):
        model = MultinomialNB()
        X_train = train_feat[dev_index]
        y_train = dependent[dev_index]
        X_valid = train_feat[val_index]
        model.fit(X_train, y_train)
        pred_train[val_index,:] = model.predict_proba(X_valid)
        pred_test += model.predict_proba(test_feat)
    pred_test = pred_test / 5.
    train_score = train_data[['id','author']].join(pd.DataFrame(pred_train))
    test_score = test_data[['id']].join(pd.DataFrame(pred_test))
    print('log loss:', metrics.log_loss(train_score['author'], train_score.iloc[:,2:]))
    return train_score, test_score

insample_word, outsample_word = naive_bayes(train_word, test_word)
insample_char_cnt, outsample_char_cnt = naive_bayes(train_char_cnt, test_char_cnt)
insample_char_tf, outsample_char_tf = naive_bayes(train_char_tf, test_char_tf)

insample_score = insample_word.merge(insample_char_cnt, on=['author','id'])
insample_score = insample_score.merge(insample_char_tf, on=['author','id'])
outsample_score = outsample_word.merge(outsample_char_cnt, on=['id'])
outsample_score = outsample_score.merge(outsample_char_tf, on=['id'])
insample_score.to_csv('../../data/spooky-author/data/train_nb_score.csv', index=False)
outsample_score.to_csv('../../data/spooky-author/data/test_nb_score.csv', index=False)

vectorizer =  TfidfVectorizer(ngram_range=(1,5), analyzer='char')
train_text = list(train_data.text.values)
test_text = list(test_data.text.values)
full_text = train_text + test_text
full_text = vectorizer.fit_transform(full_text)
svd = TruncatedSVD(n_components=10, algorithm='arpack')
svd.fit(full_text)
train_char_svd  = pd.DataFrame(svd.transform(vectorizer.transform(train_text)))
test_char_svd =  pd.DataFrame(svd.transform(vectorizer.transform(test_text)))
train_char_svd.columns = ['svd_char_' + str(x) for x in range(10)]
test_char_svd.columns = ['svd_char_' + str(x) for x in range(10)]
print('train:', train_char_svd.shape)
print('test:', test_char_svd.shape)

vectorizer =  TfidfVectorizer(stop_words='english', ngram_range=(1,3))
train_text = list(train_data.text.values)
test_text = list(test_data.text.values)
full_text = train_text + test_text
full_text = vectorizer.fit_transform(full_text)
svd = TruncatedSVD(n_components=10, algorithm='arpack')
svd.fit(full_text)
train_wrd_svd  = pd.DataFrame(svd.transform(vectorizer.transform(train_text)))
test_wrd_svd =  pd.DataFrame(svd.transform(vectorizer.transform(test_text)))
train_wrd_svd.columns = ['svd_wrd_' + str(x) for x in range(10)]
test_wrd_svd.columns = ['svd_wrd_' + str(x) for x in range(10)]
print('train:', train_wrd_svd.shape)
print('test:', test_wrd_svd.shape)

train_feats = pd.concat([train_wrd_svd, train_char_svd], axis=1)
test_feats = pd.concat([test_wrd_svd, test_char_svd], axis=1)
train_feats.to_csv('../../data/spooky-author/data/train_nb_feats.csv', index=False)
test_feats.to_csv('../../data/spooky-author/data/test_nb_feats.csv', index=False)