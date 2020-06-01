import spacy
import pandas as pd
import numpy as np
from urllib.parse import unquote
from collections import OrderedDict
import re
nlp = spacy.load('en_core_web_lg')

def domain(t):
    while not t._.subj and not t._.poss and not (t.dep_ == 'xcomp' and t.head._.obj) and t != t.head:
        t = t.head
    return t

def ccom(t):
    return [t2 for t2 in t.head._.d]

spacy.tokens.doc.Doc.set_extension(
    'to', method=lambda doc, offset: [t for t in doc if t.idx == offset][0], force=True)
spacy.tokens.token.Token.set_extension(
    'c', getter=lambda t: [c for c in t.children], force=True)
spacy.tokens.token.Token.set_extension(
    'd', getter=lambda t: [c for c in t.sent if t in list(c.ancestors)], force=True)
spacy.tokens.token.Token.set_extension(
    'subj', getter=lambda t: ([c for c in t._.c if c.dep_.startswith('nsubj')] + [False])[0], force=True)
spacy.tokens.token.Token.set_extension(
    'obj', getter=lambda t: ([c for c in t._.c if c.dep_.startswith('dobj')] + [False])[0], force=True)
spacy.tokens.token.Token.set_extension(
    'poss', getter=lambda t: ([c for c in t._.c if c.dep_.startswith('poss')] + [False])[0], force=True)
spacy.tokens.token.Token.set_extension(
    'span', method=lambda t, t2: t.doc[t.i:t2.i] if t.i < t2.i else t.doc[t2.i:t.i], force=True)
spacy.tokens.token.Token.set_extension('domain', getter=domain, force=True)
spacy.tokens.token.Token.set_extension('ccom', getter=ccom, force=True)


def applyDisq(condition, candidates, candidate_dict, debug = False):
    badnames = sum([nameset(c, candidate_dict) for c in candidates if c in condition[0]], [])
    badcands = [c for c in candidates if c.text in badnames]
    if debug and len(badcands) > 0: print('Disqualified:', badcands, '<', condition[1])
    return [c for c in candidates if c not in badcands]

def applyDisqs(conditions, candidates, candidate_dict, debug = False):
    for condition in conditions:
        if len(candidates) < 1: return candidates
        candidates = applyDisq(condition, candidates, candidate_dict, debug)
    return candidates

def disqGen(t, candidates, candidate_dict, debug = False):
    conds = [(t._.ccom,
             "disqualify candidates c-commanded by genpn; e.g. e.g. *Julia read his_i book about John_i's life."),
             ([t2 for t2 in candidates if t in t2._.ccom and t2.head.dep_ == 'appos'],
             "disqualify candidates modified by an appositive with genpn; e.g. *I wanted to see John_i, his_i father.")
            ]
    return applyDisqs(conds, candidates, candidate_dict, debug)

def disqOthers(t, candidates, candidate_dict, debug = False):
    conds = [([t2 for t2 in t._.ccom if t2.i > t.i],
             "disqualify candidates c-commanded by pn, unless they were preposed;\
             e.g. *He_i cried before John_i laughed. vs. Before John_i laughed, he_i cried."),
             ([t2 for t2 in candidates if t in t2._.ccom and t2._.domain == t._.domain
              and not (t.head.text == 'with' and t.head.head.lemma_ == 'take')],
             "disqualify candidates that c-command pn, unless in different domain;\
             e.g. Mary said that *John_i hit him_i. vs. John_i said that Mary hit him_i;\
             random hard-coded exception: `take with'"),
             ([t2 for t2 in candidates if t2._.domain.dep_ == 'xcomp' and t2._.domain.head._.obj and
               t2 == t2._.domain.head._.obj],
             "for xcomps with subjects parsed as upstairs dobj, disallow coref with that dobj;\
             e.g. *Mary wanted John_i to forgive him_i.")
            ]
    return applyDisqs(conds, candidates, candidate_dict, debug)

def disq(t, candidates, candidate_dict, debug = False):
    func = disqGen if t.dep_ == 'poss' else disqOthers
    candidates = func(t, candidates, candidate_dict, debug)
    return candidates

def find_head(w, wo, doc):
    t = False; backtrack = 0
    while not t:
        try:
            t = doc._.to(wo)
        except IndexError:
            wo -= 1; backtrack += 1
    while t.dep_ == 'compound' and t.head.idx >= wo and t.head.idx < len(w) + wo + backtrack: t = t.head
    return t

def subnames(name):
    if type(name) != str: name = candidate_dict[name]
    parts = name.split(' ')
    subnames_ = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts) + 1):
            sub = ' '.join(parts[i:j])
            if len(sub) > 2: subnames_.append(sub)
    return subnames_

def nameset(name, candidate_dict):
    if type(name) != str: name = candidate_dict[name]
    subnames_ = [sn for sn in subnames(name)]
    return [c for c in subnames_ if c not in sum([subnames(c) for c in candidate_dict.values()
                                                  if c not in subnames_ and name not in subnames(c)], [])]

def candInstances(candidates, candidate_dict):
    candidates_by_name = {}
    for c in sorted(candidates, key = lambda c: len(candidate_dict[c]), reverse = True):
        name = candidate_dict[c]
        for name2 in candidates_by_name.keys():
            if name in nameset(name2, candidate_dict): name = name2; break
        candidates_by_name[name] = candidates_by_name.get(name, []) + [c]
    return candidates_by_name

import gender_guesser.detector as gender # oops
gd = gender.Detector()

def filterGender(candidates_by_name, a, b, pn):
    badnames = []
    gender = 'female' if pn in ['She', 'she', 'her', 'Her'] else 'male'
    for name in candidates_by_name.keys():
        if a in subnames(name) or b in subnames(name): continue
        genderii = gd.get_gender(name.split(' ')[0])
        if gender == 'male' and genderii == 'female': badnames += [name]; continue
        if gender == 'female' and genderii == 'male': badnames += [name]; continue
    for name in badnames: candidates_by_name.pop(name)
    return candidates_by_name

def urlMatch(a, b, url, candidate_dict):
    url = re.sub('[^\x00-\x7F]', '*', unquote(url.split('/')[-1])).replace('_', ' ').lower()
    result = []
    result += [('a_url',(sorted([len(n.split(' ')) for n in nameset(a.lower(), candidate_dict)
                            if n in nameset(url, candidate_dict)], reverse = True) + [0])[0])]
    result += [('b_url',(sorted([len(n.split(' ')) for n in nameset(b.lower(), candidate_dict)
                              if n in nameset(url, candidate_dict)], reverse = True) + [0])[0])]
    return result

def parallel(t1, t2):
    if t1.dep_.startswith('nsubj'): return t2.dep_.startswith('nsubj')
    if t1.dep_.startswith('dobj'): return t2.dep_.startswith('dobj')
    if t1.dep_.startswith('dative'): return t2.dep_.startswith('dative')
    return False

def depthTo(t1, t2):
    depth = 0
    while t1 != t2 and t1 != t1.head:
        t1 = t1.head
        depth += 1
    return depth

def nodeDist(t1, t2):
    if t1 == t2: return 0
    if t2 in t1._.d: return depthTo(t2, t1)
    if t1 in t2._.d: return depthTo(t1, t2)
    t = t1
    while t1 not in t._.d or t2 not in t._.d and t != t.head: t = t.head
    return depthTo(t1, t) + depthTo(t2, t)

def synDist(t, pn, doc, debug = False):
    doc_sents = list(doc.sents)
    sspan = doc_sents.index(pn.sent) - doc_sents.index(t.sent)
    if sspan == 0:
        dist = nodeDist(t, pn)
    else:
        dist = nodeDist(pn, doc_sents[doc_sents.index(pn.sent)].root) + \
               nodeDist(t, doc_sents[doc_sents.index(t.sent)].root)
    if debug:
        print('pn dist:', nodeDist(pn, doc_sents[doc_sents.index(pn.sent)].root), '; t dist:',
              nodeDist(t, doc_sents[doc_sents.index(t.sent)].root), '; span:', sspan)
    sspan = abs(sspan) * 1 if sspan >= 0 else abs(sspan) * 1.3
    return dist + sspan

def charDist(t1, t2):
    if t2.idx > t1.idx:
        return t2.idx - t1.idx + len(t1.text)
    else:
        return (t1.idx - t2.idx + len(t2.text)) * 1.3

def thetaProminence(t, mult = 1, debug = False):
    while t.dep_ == 'compound': t = t.head
    if debug: print('t dep_:', t.dep_)
    if t.dep_ == 'pobj': mult = 1.3 if t.head.i < t.head.head.i else 1
    if t._.domain.dep_ == 'advcl': mult = 1.3 if t.head.i < t._.domain.head.i else 1
    if t.dep_.startswith('nsubj'): score = 1
    elif t.dep_.startswith('dobj'): score = 0.8
    elif t.dep_.startswith('dative'): score = 0.6
    elif t.dep_.startswith('pobj'): score = 0.4
    elif t.dep_.startswith('poss'): score = 0.3
    else: score = 0.1
    if debug: print('mult:', mult, '; score:', score)
    return min(1, score * mult)

def score(label, candidates_by_name, a_cand, b_cand, func, minsc = None, method = 'sum'):
    if method == 'sum':
        scores = {name: sum([func(t) for t in tokens]) for name, tokens in candidates_by_name.items()}
    elif method == 'meandiff':
        mean = np.mean(sum([[func(t) for t in tokens] for tokens in candidates_by_name.values()], []))
        scores = {name: mean - min([func(t) for t in tokens]) for name, tokens in candidates_by_name.items()}
    sca = scores[a_cand] if a_cand else minsc
    scb = scores[b_cand] if b_cand else minsc
    screst = [v for n, v in scores.items() if n != a_cand and n != b_cand]
    if method == 'sum':
        screst = sum(screst) if len(screst) > 0 else minsc
    elif method == 'meandiff':
        screst = max(screst) if len(screst) > 0 else minsc
    result = []
    result += [('a_' + label, sca)]
    result += [('b_' + label, scb)]
    result += [('n_' + label, screst)]
    return result

def load_row(data, i):
    return tuple(data.iloc[i])

def annotateSet(data, minsc = None, debug = False):
    annotated_data = pd.DataFrame()
    row_batch = []

    for i in range(annotated_data.shape[0], data.shape[0]):
        id, text, pn, pno, a, ao, ag, b, bo, bg, url = load_row(data, i)
        doc = nlp(text)
        pnt, at, bt = (doc._.to(pno), find_head(a, ao, doc), find_head(b, bo, doc))
        candidate_dict = {e.root: re.sub('\'s$', '', e.text)
                          for e in [e for e in doc.ents if e.root.ent_type_ == 'PERSON']}
        candidate_dict.update({c.root: re.sub('\'s$', '', c.text)
                               for c in doc.noun_chunks if c.root.pos_ == 'PROPN'
                               and c.text in sum([subnames(n) for n in candidate_dict.values()], [])
                               and c.root not in candidate_dict.keys()})
        candidate_dict.update({t: w for t, w in [(at, a), (bt, b)]})
        candidates = disq(pnt, list(candidate_dict.keys()), candidate_dict, debug = False)
        candidates_by_name = candInstances(candidates, candidate_dict)
        candidates_by_name = filterGender(candidates_by_name, a, b, pn)
        a_cand = ([name for name, tokens in candidates_by_name.items() if at in tokens] + [False])[0]
        b_cand = ([name for name, tokens in candidates_by_name.items() if bt in tokens] + [False])[0]
        features = OrderedDict({})
        features.update([('ID',id)])
        features.update([('a_out', 0 if a_cand else 1), ('b_out',0 if b_cand else 1)])
        features.update(urlMatch(a, b, url, candidate_dict))
        features.update([('a_cc', 1 if a_cand and pnt in at._.ccom else 0),
                         ('b_cc', 1 if b_cand and pnt in bt._.ccom else 0)])
        features.update(score('par', candidates_by_name, a_cand, b_cand, lambda t: parallel(t, pnt), minsc = minsc))
        features.update(score('th', candidates_by_name, a_cand, b_cand, thetaProminence, minsc = minsc))
        features.update(score('loc', candidates_by_name, a_cand, b_cand, lambda t: synDist(t, pnt, doc),
                              method='meandiff', minsc = minsc))
        features.update({'n_cands': len(candidates_by_name)})
        features.update(score('cloc', candidates_by_name, a_cand, b_cand, lambda t: charDist(t, pnt),
                              method='meandiff', minsc = minsc))
        if debug: print(id, '>', 'a:', 1 if ag else 0, 'b:', 1 if bg else 0, features)
        row_batch += [features]
    if annotated_data.shape[0] != data.shape[0]:
        annotated_data = annotated_data.append(row_batch, ignore_index = True)
    annotated_data['a_cloc'] = annotated_data['a_cloc'] / 100
    annotated_data['b_cloc'] = annotated_data['b_cloc'] / 100
    annotated_data['n_cloc'] = annotated_data['n_cloc'] / 100
    return annotated_data

