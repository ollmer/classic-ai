from flask import Flask, request, jsonify
import os
import sys
import copy
import string
import numpy as np
from utils import Phonetic, PoemTemplateLoader, Word2vecProcessor

app = Flask(__name__)

import warnings
warnings.filterwarnings("ignore")

DATASETS_PATH = 'data'
template_loader = PoemTemplateLoader(os.path.join(DATASETS_PATH, 'classic_poems.json'))
word2vec = Word2vecProcessor(os.path.join(DATASETS_PATH, 'web_upos_cbow_300_20_2017.bin.gz'))

new_forms = {}
word_forms = {}
with open(os.path.join(DATASETS_PATH, 'word_stress_pos_full.txt')) as f:
    for l in f:
        l = l.strip()
        word, key = l.split(',', 1)
        if key not in new_forms:
            new_forms[key] = []
        word_forms[word] = key
        new_forms[key].append(word)

def sound_distance(word1, word2, suffix_len=3):    
    suffix1 = (' ' * suffix_len + word1)[-suffix_len:]
    suffix2 = (' ' * suffix_len + word2)[-suffix_len:]
    distance = sum((ch1 != ch2) for ch1, ch2 in zip(suffix1, suffix2))
    return distance        

def generate_poem(seed, poet_id):
    skip = ['мой','мою','моя','мое','мне','моё','мной','тот','для','вот','все']

    # выбираем шаблон на основе случайного стихотворения из корпуса
    template = template_loader.get_random_template(poet_id)
    skip_poem = any([len(l) < 2 for l in template]) or template[0][0] in string.ascii_letters
    while skip_poem:
        template = template_loader.get_random_template(poet_id)
        skip_poem = any([len(l) < 2 for l in template]) or template[0][0] in string.ascii_letters

    poem = copy.deepcopy(template)

    # оцениваем word2vec-вектор темы
    seed_vec = word2vec.text_vector(seed)

    used = set()
    replaced = 0
    total = 0
    skip_poem = any([len(l) < 2 for l in poem])
    # заменяем слова в шаблоне на более релевантные теме
    for li, line in enumerate(poem):
        llen = len(line) - 1
        for ti, token in enumerate(line):
            total += 1
            if not token.isalpha():
                continue

            word = token.lower()
            if len(word) < 3 or word[:3] == 'как' or word in skip:
                continue

            # выбираем слова - кандидаты на замену: максимально похожие фонетически на исходное слово
            if word in word_forms:
                form = word_forms[word]
            else:
                continue
#             form = phonetic.get_form(token)
            candidate_phonetic_distances = [
                (replacement_word, sound_distance(replacement_word, word))
                for replacement_word in new_forms[form]
                ]
            if not candidate_phonetic_distances:
                continue
            if ti == llen:
                min_phonetic_distance = min(d for w, d in candidate_phonetic_distances)
                replacement_candidates = [w for w, d in candidate_phonetic_distances 
                                          if d == min_phonetic_distance and w not in used]
            else:
                replacement_candidates = [w for w, d in candidate_phonetic_distances if w not in used]             
                
            # из кандидатов берем максимально близкое теме слово
            word2vec_distances = [
                (replacement_word, word2vec.distance(seed_vec, word2vec.word_vector(replacement_word)))
                for replacement_word in replacement_candidates
                ]
            word2vec_distances.sort(key=lambda pair: pair[1])
            if not word2vec_distances:
                continue
            word2vec_nearest = [k for k,v in word2vec_distances[:3]]
            new_word = word2vec_nearest[0] # np.random.choice(word2vec_nearest)
            
            if poem[li][ti] != new_word:
                poem[li][ti] = new_word
                replaced += 1
                used.add(new_word)

    # собираем получившееся стихотворение из слов
    generated_poem = '\n'.join([' '.join([token for token in line]).capitalize() for line in poem])
    return generated_poem
        
@app.route('/ready')
def ready():
    return 'OK'


@app.route('/generate/<poet_id>', methods=['POST'])
def generate(poet_id):
    request_data = request.get_json()
    seed = request_data['seed']
    generated_poem = generate_poem(seed, poet_id)
    return jsonify({'poem': generated_poem})


if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(host='0.0.0.0', port=8000)