import pandas as pd
import numpy as np
import re

rnn_1 = pd.read_csv('../data/model/simple_stack.csv')
rnn_2 = pd.read_csv('../data/model/simple_stack_1.csv')
rnn_3 = pd.read_csv('../data/other/model/simple_stack.csv')
rnn = rnn_1.append(rnn_2).append(rnn_3)
rnn = rnn.groupby('id').mean().reset_index()
rnn.columns = ['id'] + ['rnn_' + x for x in rnn.columns[1:]]
nbsvm = pd.read_csv('../data/model/nbsvm.csv')
nbsvm.columns = ['id'] + ['nbsvm_' + x for x in nbsvm.columns[1:]]
logreg = pd.read_csv('../data/model/ftrl.csv')
logreg.columns = ['id'] + ['logreg_' + x for x in logreg.columns[1:]]
ftrl = pd.read_csv('../data/other/model/word-batch-oof.csv')
feats = ['id'] + [x for x in ftrl.columns if 'oof' in x]
ftrl = ftrl[feats]
ftrl.columns = ['id'] + ['ftrl_' + x for x in rnn_1.columns[1:]]

stack = rnn.merge(nbsvm, on='id').merge(logreg, on='id').merge(ftrl, on='id')
stack.to_csv('../data/train_stack_scores.csv', index=False)
stack[[x for x in stack.columns if 'obscene' in x]].corr()

rnn_1 = pd.read_csv('../data/submit/simple_stack.csv')
rnn_2 = pd.read_csv('../data/submit/simple_stack_1.csv')
rnn_3 = pd.read_csv('../data/other/submit/simple_stack.csv')
rnn = rnn_1.append(rnn_2).append(rnn_3)
rnn = rnn.groupby('id').mean().reset_index()
rnn.columns = ['id'] + ['rnn_' + x for x in rnn.columns[1:]]
nbsvm = pd.read_csv('../data/submit/nbsvm.csv')
nbsvm.columns = ['id'] + ['nbsvm_' + x for x in nbsvm.columns[1:]]
logreg = pd.read_csv('../data/submit/ftrl.csv')
logreg.columns = ['id'] + ['logreg_' + x for x in logreg.columns[1:]]
ftrl = pd.read_csv('../data/other/submit/word-batch.csv')
ftrl.columns = ['id'] + ['ftrl_' + x for x in rnn_1.columns[1:]]

stack = rnn.merge(nbsvm, on='id').merge(logreg, on='id').merge(ftrl, on='id')
stack.to_csv('../data/test_stack_scores.csv', index=False)
stack[[x for x in stack.columns if 'obscene' in x]].corr()