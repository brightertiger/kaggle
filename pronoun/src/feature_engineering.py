import pandas as pd
import numpy as np
import spacy
import re
from urllib.parse import unquote
from typing import Dict, Any, List, Tuple
import gender_guesser.detector as gender

nlp = spacy.load('en_core_web_lg')
gender_detector = gender.Detector()

class FeatureExtractor:
    def __init__(self):
        self._setup_spacy_extensions()
    
    def _setup_spacy_extensions(self):
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
        spacy.tokens.token.Token.set_extension('domain', getter=self._get_domain, force=True)
        spacy.tokens.token.Token.set_extension('ccom', getter=self._get_ccom, force=True)
    
    def _get_domain(self, token):
        t = token
        while not t._.subj and not t._.poss and not (t.dep_ == 'xcomp' and t.head._.obj) and t != t.head:
            t = t.head
        return t
    
    def _get_ccom(self, token):
        return [t2 for t2 in token.head._.d]
    
    def extract_basic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()
        
        data['URL'] = data['URL'].map(lambda x: x.replace('http://en.wikipedia.org/wiki/', ''))
        data['URL'] = data['URL'].map(lambda x: x.replace('_', ' '))
        
        data['has_a'] = data[['A', 'URL']].apply(lambda x: self._url_contains_name(*x), axis=1).astype(int)
        data['has_b'] = data[['B', 'URL']].apply(lambda x: self._url_contains_name(*x), axis=1).astype(int)
        
        data['index_p'] = data[['Text', 'Pronoun', 'Pronoun-offset']].apply(lambda x: self._word_index(*x), axis=1)
        data['index_a'] = data[['Text', 'A', 'A-offset']].apply(lambda x: self._word_index(*x), axis=1)
        data['index_b'] = data[['Text', 'B', 'B-offset']].apply(lambda x: self._word_index(*x), axis=1)
        
        data['dist_a'] = abs(data['index_p'] - data['index_a']).apply(lambda x: self._bin_distance(x))
        data['dist_b'] = abs(data['index_p'] - data['index_b']).apply(lambda x: self._bin_distance(x))
        
        data = data.drop(['index_a', 'index_b', 'index_p'], axis=1)
        
        data['name_a'] = data[['Text', 'A']].apply(lambda x: self._name_count(*x), axis=1)
        data['name_b'] = data[['Text', 'B']].apply(lambda x: self._name_count(*x), axis=1)
        
        return data
    
    def _url_contains_name(self, name: str, url: str) -> bool:
        name_words = name.lower().split()
        url_words = url.lower().split()
        return any(word in url_words for word in name_words)
    
    def _word_index(self, text: str, word: str, offset: int) -> int:
        doc = nlp(text)
        tokens = [(token.idx, token) for token in doc]
        word_first = word.split()[0]
        
        for idx, (pos, token) in enumerate(tokens):
            if str(token) == word_first and abs(pos - offset) < 10:
                return idx
        return 0
    
    def _bin_distance(self, value: int) -> int:
        if value < 3: return 1
        elif value < 4: return 2
        elif value < 5: return 3
        elif value < 8: return 4
        elif value < 12: return 5
        elif value < 16: return 6
        elif value < 24: return 7
        elif value < 32: return 8
        else: return 9
    
    def _name_count(self, text: str, word: str) -> int:
        try:
            count_total = len(re.findall(word, text))
            text_words = text.split()
            word_parts = word.split()
            count_first = text_words.count(word_parts[0])
            count_last = text_words.count(word_parts[-1])
            return count_first + count_last - count_total
        except:
            return 0
    
    def extract_linguistic_features(self, data: pd.DataFrame) -> pd.DataFrame:
        features = []
        for _, row in data.iterrows():
            features.append(self._extract_row_features(row))
        
        feature_df = pd.DataFrame(features)
        return pd.merge(data, feature_df, on='ID')
    
    def _extract_row_features(self, row: pd.Series) -> Dict[str, Any]:
        doc = nlp(row['Text'])
        pronoun_token = doc._.to(row['Pronoun-offset'])
        name_a_token = self._find_head(row['A'], row['A-offset'], doc)
        name_b_token = self._find_head(row['B'], row['B-offset'], doc)
        
        candidate_dict = self._build_candidate_dict(doc, name_a_token, name_b_token, row['A'], row['B'])
        candidates = self._filter_candidates(pronoun_token, list(candidate_dict.keys()), candidate_dict)
        candidates_by_name = self._group_candidates_by_name(candidates, candidate_dict)
        candidates_by_name = self._filter_by_gender(candidates_by_name, row['A'], row['B'], row['Pronoun'])
        
        features = self._compute_features(pronoun_token, name_a_token, name_b_token, 
                                        candidates_by_name, candidate_dict, row)
        
        return features
    
    def _find_head(self, word: str, offset: int, doc) -> Any:
        token = None
        backtrack = 0
        while not token:
            try:
                token = doc._.to(offset)
            except IndexError:
                offset -= 1
                backtrack += 1
        
        while (token.dep_ == 'compound' and 
               token.head.idx >= offset and 
               token.head.idx < len(word) + offset + backtrack):
            token = token.head
        return token
    
    def _build_candidate_dict(self, doc, name_a_token, name_b_token, name_a, name_b) -> Dict:
        candidate_dict = {}
        
        for entity in doc.ents:
            if entity.root.ent_type_ == 'PERSON':
                candidate_dict[entity.root] = re.sub(r"'s$", '', entity.text)
        
        for chunk in doc.noun_chunks:
            if (chunk.root.pos_ == 'PROPN' and 
                chunk.root not in candidate_dict.keys()):
                candidate_dict[chunk.root] = re.sub(r"'s$", '', chunk.text)
        
        candidate_dict[name_a_token] = name_a
        candidate_dict[name_b_token] = name_b
        
        return candidate_dict
    
    def _filter_candidates(self, pronoun_token, candidates, candidate_dict) -> List:
        if pronoun_token.dep_ == 'poss':
            return self._filter_genitive_candidates(pronoun_token, candidates, candidate_dict)
        else:
            return self._filter_other_candidates(pronoun_token, candidates, candidate_dict)
    
    def _filter_genitive_candidates(self, pronoun_token, candidates, candidate_dict) -> List:
        disqualified = []
        for candidate in candidates:
            if candidate in pronoun_token._.ccom:
                disqualified.append(candidate)
            if (pronoun_token in candidate._.ccom and 
                candidate.head.dep_ == 'appos'):
                disqualified.append(candidate)
        
        return [c for c in candidates if c not in disqualified]
    
    def _filter_other_candidates(self, pronoun_token, candidates, candidate_dict) -> List:
        disqualified = []
        
        for candidate in candidates:
            if (candidate in pronoun_token._.ccom and 
                candidate.i > pronoun_token.i):
                disqualified.append(candidate)
            
            if (pronoun_token in candidate._.ccom and 
                candidate._.domain == pronoun_token._.domain and
                not (pronoun_token.head.text == 'with' and 
                     pronoun_token.head.head.lemma_ == 'take')):
                disqualified.append(candidate)
        
        return [c for c in candidates if c not in disqualified]
    
    def _group_candidates_by_name(self, candidates, candidate_dict) -> Dict:
        candidates_by_name = {}
        for candidate in sorted(candidates, key=lambda c: len(candidate_dict[c]), reverse=True):
            name = candidate_dict[candidate]
            for existing_name in candidates_by_name.keys():
                if self._is_subname(name, existing_name):
                    name = existing_name
                    break
            candidates_by_name[name] = candidates_by_name.get(name, []) + [candidate]
        
        return candidates_by_name
    
    def _is_subname(self, name1: str, name2: str) -> bool:
        return name1 in self._get_subnames(name2)
    
    def _get_subnames(self, name: str) -> List[str]:
        parts = name.split(' ')
        subnames = []
        for i in range(len(parts)):
            for j in range(i + 1, len(parts) + 1):
                subname = ' '.join(parts[i:j])
                if len(subname) > 2:
                    subnames.append(subname)
        return subnames
    
    def _filter_by_gender(self, candidates_by_name, name_a, name_b, pronoun) -> Dict:
        pronoun_gender = 'female' if pronoun.lower() in ['she', 'her'] else 'male'
        filtered_candidates = {}
        
        for name, tokens in candidates_by_name.items():
            if name_a in self._get_subnames(name) or name_b in self._get_subnames(name):
                filtered_candidates[name] = tokens
                continue
            
            first_name = name.split(' ')[0]
            detected_gender = gender_detector.get_gender(first_name)
            
            if ((pronoun_gender == 'male' and detected_gender == 'female') or
                (pronoun_gender == 'female' and detected_gender == 'male')):
                continue
            
            filtered_candidates[name] = tokens
        
        return filtered_candidates
    
    def _compute_features(self, pronoun_token, name_a_token, name_b_token, 
                         candidates_by_name, candidate_dict, row) -> Dict[str, Any]:
        features = {'ID': row['ID']}
        
        name_a_candidate = self._find_candidate_for_token(name_a_token, candidates_by_name)
        name_b_candidate = self._find_candidate_for_token(name_b_token, candidates_by_name)
        
        features['a_out'] = 0 if name_a_candidate else 1
        features['b_out'] = 0 if name_b_candidate else 1
        
        url_features = self._compute_url_features(row['A'], row['B'], row['URL'], candidate_dict)
        features.update(url_features)
        
        features['a_cc'] = 1 if name_a_candidate and pronoun_token in name_a_token._.ccom else 0
        features['b_cc'] = 1 if name_b_candidate and pronoun_token in name_b_token._.ccom else 0
        
        syntactic_features = self._compute_syntactic_features(
            pronoun_token, candidates_by_name, name_a_candidate, name_b_candidate, row['Text'])
        features.update(syntactic_features)
        
        features['n_cands'] = len(candidates_by_name)
        
        return features
    
    def _find_candidate_for_token(self, token, candidates_by_name) -> str:
        for name, tokens in candidates_by_name.items():
            if token in tokens:
                return name
        return None
    
    def _compute_url_features(self, name_a, name_b, url, candidate_dict) -> Dict[str, Any]:
        url_clean = re.sub(r'[^\x00-\x7F]', '*', unquote(url.split('/')[-1])).replace('_', ' ').lower()
        
        a_url_score = self._compute_url_match_score(name_a.lower(), url_clean, candidate_dict)
        b_url_score = self._compute_url_match_score(name_b.lower(), url_clean, candidate_dict)
        
        return {'a_url': a_url_score, 'b_url': b_url_score}
    
    def _compute_url_match_score(self, name, url, candidate_dict) -> int:
        name_subnames = self._get_subnames(name)
        url_subnames = self._get_subnames(url)
        
        matches = [len(n.split(' ')) for n in name_subnames if n in url_subnames]
        return max(matches) if matches else 0
    
    def _compute_syntactic_features(self, pronoun_token, candidates_by_name, 
                                   name_a_candidate, name_b_candidate, text) -> Dict[str, Any]:
        features = {}
        
        doc = nlp(text)
        
        parallel_scores = self._compute_parallel_scores(pronoun_token, candidates_by_name)
        theta_scores = self._compute_theta_scores(pronoun_token, candidates_by_name)
        location_scores = self._compute_location_scores(pronoun_token, candidates_by_name, doc)
        char_distance_scores = self._compute_char_distance_scores(pronoun_token, candidates_by_name)
        
        features.update(self._format_scores('par', parallel_scores, name_a_candidate, name_b_candidate))
        features.update(self._format_scores('th', theta_scores, name_a_candidate, name_b_candidate))
        features.update(self._format_scores('loc', location_scores, name_a_candidate, name_b_candidate))
        features.update(self._format_scores('cloc', char_distance_scores, name_a_candidate, name_b_candidate))
        
        return features
    
    def _compute_parallel_scores(self, pronoun_token, candidates_by_name) -> Dict[str, float]:
        scores = {}
        for name, tokens in candidates_by_name.items():
            parallel_count = sum(1 for token in tokens if self._is_parallel(token, pronoun_token))
            scores[name] = parallel_count
        return scores
    
    def _is_parallel(self, token1, token2) -> bool:
        if token1.dep_.startswith('nsubj'): return token2.dep_.startswith('nsubj')
        if token1.dep_.startswith('dobj'): return token2.dep_.startswith('dobj')
        if token1.dep_.startswith('dative'): return token2.dep_.startswith('dative')
        return False
    
    def _compute_theta_scores(self, pronoun_token, candidates_by_name) -> Dict[str, float]:
        scores = {}
        for name, tokens in candidates_by_name.items():
            theta_sum = sum(self._compute_theta_prominence(token) for token in tokens)
            scores[name] = theta_sum
        return scores
    
    def _compute_theta_prominence(self, token) -> float:
        while token.dep_ == 'compound':
            token = token.head
        
        if token.dep_.startswith('nsubj'): return 1.0
        elif token.dep_.startswith('dobj'): return 0.8
        elif token.dep_.startswith('dative'): return 0.6
        elif token.dep_.startswith('pobj'): return 0.4
        elif token.dep_.startswith('poss'): return 0.3
        else: return 0.1
    
    def _compute_location_scores(self, pronoun_token, candidates_by_name, doc) -> Dict[str, float]:
        scores = {}
        all_distances = []
        
        for name, tokens in candidates_by_name.items():
            distances = [self._compute_syntactic_distance(token, pronoun_token, doc) for token in tokens]
            all_distances.extend(distances)
        
        mean_distance = np.mean(all_distances) if all_distances else 0
        
        for name, tokens in candidates_by_name.items():
            distances = [self._compute_syntactic_distance(token, pronoun_token, doc) for token in tokens]
            scores[name] = mean_distance - min(distances) if distances else 0
        
        return scores
    
    def _compute_syntactic_distance(self, token, pronoun_token, doc) -> float:
        doc_sents = list(doc.sents)
        sent_span = doc_sents.index(pronoun_token.sent) - doc_sents.index(token.sent)
        
        if sent_span == 0:
            distance = self._node_distance(token, pronoun_token)
        else:
            distance = (self._node_distance(pronoun_token, doc_sents[doc_sents.index(pronoun_token.sent)].root) +
                       self._node_distance(token, doc_sents[doc_sents.index(token.sent)].root))
        
        sent_span = abs(sent_span) * 1 if sent_span >= 0 else abs(sent_span) * 1.3
        return distance + sent_span
    
    def _node_distance(self, token1, token2) -> int:
        if token1 == token2: return 0
        if token2 in token1._.d: return self._depth_to(token2, token1)
        if token1 in token2._.d: return self._depth_to(token1, token2)
        
        common_ancestor = token1
        while (token1 not in common_ancestor._.d or token2 not in common_ancestor._.d) and common_ancestor != common_ancestor.head:
            common_ancestor = common_ancestor.head
        
        return self._depth_to(token1, common_ancestor) + self._depth_to(token2, common_ancestor)
    
    def _depth_to(self, from_token, to_token) -> int:
        depth = 0
        while from_token != to_token and from_token != from_token.head:
            from_token = from_token.head
            depth += 1
        return depth
    
    def _compute_char_distance_scores(self, pronoun_token, candidates_by_name) -> Dict[str, float]:
        scores = {}
        all_distances = []
        
        for name, tokens in candidates_by_name.items():
            distances = [self._char_distance(token, pronoun_token) for token in tokens]
            all_distances.extend(distances)
        
        mean_distance = np.mean(all_distances) if all_distances else 0
        
        for name, tokens in candidates_by_name.items():
            distances = [self._char_distance(token, pronoun_token) for token in tokens]
            scores[name] = mean_distance - min(distances) if distances else 0
        
        return scores
    
    def _char_distance(self, token1, token2) -> float:
        if token2.idx > token1.idx:
            return token2.idx - token1.idx + len(token1.text)
        else:
            return (token1.idx - token2.idx + len(token2.text)) * 1.3
    
    def _format_scores(self, prefix, scores, name_a_candidate, name_b_candidate) -> Dict[str, Any]:
        formatted = {}
        formatted[f'a_{prefix}'] = scores.get(name_a_candidate, 0)
        formatted[f'b_{prefix}'] = scores.get(name_b_candidate, 0)
        
        other_scores = [score for name, score in scores.items() 
                       if name != name_a_candidate and name != name_b_candidate]
        formatted[f'n_{prefix}'] = max(other_scores) if other_scores else 0
        
        return formatted
