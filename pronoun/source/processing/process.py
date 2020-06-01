import pandas as pd
import numpy as np

def textHelper(row):
    tags = []
    tags += [(row['A-offset'], " [A] ")]
    tags += [(row['B-offset'], " [B] ")]
    tags += [(row['Pronoun-offset'], " [P] ")]
    tags = sorted(tags, key=lambda x : x[0], reverse=True)
    text = row["Text"]
    for offset, tag in tags:
        text = text[:offset] + tag + text[offset:]
    return text

def idxLabel(row):
    row = list(row)
    if row[0] == 1:
        return 0
    elif row[1] == 1:
        return 1
    else:
        return 2

def addTags(data, has_label=False):
    data['Text'] = data.apply(lambda x: textHelper(x), axis=1)
    if not has_label:
        return data[['Text']], None
    else:
        data['Label'] = data[['A-coref','B-coref']].apply(lambda x : idxLabel(x), axis=1)
        return data[['Text']], data[['Label']].astype(int)
