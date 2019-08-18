import bz2
import csv
import json
import random
import collections
import itertools
import string

import numpy as np
from scipy.spatial.distance import cosine

from nltk.tokenize import word_tokenize
from gensim.models import KeyedVectors
from pymystem3 import Mystem


class Phonetic(object):
    """Объект для работы с фонетическими формами слова"""

    def __init__(self, accent_file, vowels='уеыаоэёяию'):
        self.vowels = vowels
        with bz2.BZ2File(accent_file) as fin:
            self.accents_dict = json.load(fin)

    def syllables_count(self, word):
        """Количество гласных букв (слогов) в слове"""
        return sum((ch in self.vowels) for ch in word)

    def accent_syllable(self, word):
        """Номер ударного слога в слове"""
        default_accent = (self.syllables_count(word) + 1) // 2
        return self.accents_dict.get(word, default_accent)

    def get_form(self, word):
        word_syllables = self.syllables_count(word)
        word_accent = self.accent_syllable(word)
        return (word_syllables, word_accent)

    def sound_distance(self, word1, word2):
        """Фонетическое растояние на основе расстояния Левенштейна по окончаниям
        (число несовпадающих символов на соответствующих позициях)"""
        suffix_len = 3
        suffix1 = (' ' * suffix_len + word1)[-suffix_len:]
        suffix2 = (' ' * suffix_len + word2)[-suffix_len:]

        distance = sum((ch1 != ch2) for ch1, ch2 in zip(suffix1, suffix2))
        return distance

    def form_dictionary_from_csv(self, corpora_file, column='paragraph', max_docs=30000):
        """Загрузить словарь слов из CSV файла с текстами, индексированный по формам слова.
        Возвращает словарь вида:
            {форма: {множество, слов, кандидатов, ...}}
            форма — (<число_слогов>, <номер_ударного>)
        """

        corpora_tokens = []
        with open(corpora_file) as fin:
            reader = csv.DictReader(fin)
            for row in itertools.islice(reader, max_docs):
                paragraph = row[column]
                paragraph_tokens = word_tokenize(paragraph.lower())
                corpora_tokens += paragraph_tokens

        word_by_form = collections.defaultdict(set)
        for token in corpora_tokens:
            if token.isalpha():
                word_syllables = self.syllables_count(token)
                word_accent = self.accent_syllable(token)
                form = (word_syllables, word_accent)
                word_by_form[form].add(token)

        return word_by_form

def bad_poem(template):
    return any([len(l) < 2 for l in template]) \
        or len(template) < 3 \
        or template[0][0][0] in string.ascii_letters \
        or template[1][0][0] in string.ascii_letters \
        or template[1][0][0] in string.ascii_letters \
        or template[-1][0][0] in string.ascii_letters \
        or template[-1][-1][-1] in string.ascii_letters

class PoemTemplateLoader(object):

    def __init__(self, poems_file, embeddings, min_lines=3, max_lines=8, max_string_len=120):
        self.embeddings = embeddings
        self.poet_templates = collections.defaultdict(list)
        self.min_lines = min_lines
        self.max_lines = max_lines
        self.max_string_len = max_string_len

        self.load_poems(poems_file)

    def load_poems(self, poems_file):
        with open(poems_file) as fin:
            poems = json.load(fin)

        for poet,pt in poems.items():
            for poem, vec in pt:
                template = self.poem_to_template(poem)
                if not template:
                    continue
                vec = np.asarray(vec)
                self.poet_templates[poet].append((template, vec))

    def poem_to_template(self, poem_text):
        poem_text = poem_text.lower()
        if '&' in poem_text or 'госиздат' in poem_text or 'те, те и те' in poem_text or 'гнедич' in poem_text:
            return None
        poem_text = poem_text.replace('*', '').replace('—', '-')
        poem_lines = poem_text.split('\n')
        poem_template = []
        for line in poem_lines:
            if len(line) > self.max_string_len:
                raise Exception('long line ' + poem_text)
            line_tokens = word_tokenize(line)
            line_tokens = [t for t in line_tokens if t not in ['...', '*', '…', ']', '[', ')', '(']]
            poem_template.append(line_tokens)
        return poem_template

    def get_random_template(self, poet_id, template_id=None):
        if not self.poet_templates[poet_id]:
            raise KeyError('Unknown poet "%s"' % poet_id)
        if not template_id:
            template_id = np.random.randint(0, len(self.poet_templates[poet_id]))
        poem, pvec = self.poet_templates[poet_id][template_id]
        return template_id, poem, pvec
    
    def get_nearest_template(self, poet_id, seed_vec):
        dists = []
        for poem, pvec in self.poet_templates[poet_id]:
            dists.append((poem, self.embeddings.distance(seed_vec, pvec)))
        dists.sort(key=lambda pair: pair[1])
        top = [d[0] for d in dists[:20] if not bad_poem(d[0])]
        topn = min(len(top), 5)
        n = np.random.randint(0,topn)
        print('top', len(top), n, [d[1] for d in dists[:topn]])
        return 0, top[n]
    

class Word2vecProcessor(object):
    """Объект для работы с моделью word2vec сходства слов"""

    def __init__(self, w2v_model_file):
        self.mystem = Mystem()
        self.word2vec = KeyedVectors.load_word2vec_format(w2v_model_file, binary=True)
        self.lemma2word = {word.split('_')[0]: word for word in self.word2vec.index2word}

    def word_vector(self, word):
        lemma = self.mystem.lemmatize(word)[0]
        word = self.lemma2word.get(lemma)
        return self.word2vec[word] if word in self.word2vec else None

    def text_vector(self, text):
        """Вектор текста, получается путем усреднения векторов всех слов в тексте"""
        word_vectors = [
            self.word_vector(token)
            for token in word_tokenize(text.lower())
            if token.isalpha()
            ]
        word_vectors = [vec for vec in word_vectors if vec is not None]
        return np.mean(word_vectors, axis=0)

    def distance(self, vec1, vec2):
        if vec1 is None or vec2 is None:
            return 2
        return cosine(vec1, vec2)