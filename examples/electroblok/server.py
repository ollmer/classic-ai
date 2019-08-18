from flask import Flask, request, jsonify
import os
import sys
import copy
import string
import requests
import numpy as np
from utils import Phonetic, PoemTemplateLoader, Word2vecProcessor
print('starting...')
app = Flask(__name__)

import warnings
warnings.filterwarnings("ignore")

DATASETS_PATH = 'data'

word2vec = Word2vecProcessor(os.path.join(DATASETS_PATH, 'web_upos_cbow_300_20_2017.bin.gz'))
template_loader = PoemTemplateLoader('data/vecpoems.json', word2vec)
print([(p,len(pt)) for p,pt in template_loader.poet_templates.items()])

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

def sound_distance(word1, word2, suffix_len=4):
    suffix1 = (' ' * suffix_len + word1)[-suffix_len:]
    suffix2 = (' ' * suffix_len + word2)[-suffix_len:]
    distance = sum((ch1 != ch2) for ch1, ch2 in zip(suffix1, suffix2))
    return distance

def bad_poem(template):
    return any([len(l) < 2 for l in template]) \
        or template[0][0][0] in string.ascii_letters \
        or template[-1][-1][-1] in string.ascii_letters

def generate_poem(seed, poet_id, template_id=None, emb=word2vec):
    skip = ['мой','мою','моя','мое','мне','моё','мной','тот','для','вот','все','ночи','бог','под', 'что']
    
    # оцениваем word-вектор темы
    seed_vec = emb.text_vector(seed)
    
    # выбираем шаблон на основе случайного стихотворения из корпуса
#     tid, template, pvec = template_loader.get_random_template(poet_id, template_id)
#     while bad_poem(template):
#         tid, template = template_loader.get_random_template(poet_id)
    tid, template = template_loader.get_nearest_template(poet_id, seed_vec)
#     print(tid, template)
    poem = copy.deepcopy(template)


    used = set()
    replaced = 0
    total = 0
    skip_poem = any([len(l) < 2 for l in poem])
    # заменяем слова в шаблоне на более релевантные теме
    for li, line in enumerate(poem):
        llen = len(line) - 1
        if line[-1] in string.punctuation:
            llen -= 1
        for ti, token in enumerate(line):
            total += 1

            word = token.lower().replace('*', '')
            if len(word) < 2 or word[:3] == 'как' or word in skip:
                continue

            # выбираем слова - кандидаты на замену: максимально похожие фонетически на исходное слово
            if word in word_forms:
                form = word_forms[word]
            else:
#                 print('no form', word)
                continue
            candidate_phonetic_distances = [
                (replacement_word, sound_distance(replacement_word, word))
                for replacement_word in new_forms[form] if replacement_word[:4] != 'член' 
                and replacement_word not in ['попа','попу','попе','попой']
                ]
            if not candidate_phonetic_distances:
#                 print('no near', word)
                continue
            if ti == llen:
                min_phonetic_distance = min(d for w, d in candidate_phonetic_distances)
                replacement_candidates = [w for w, d in candidate_phonetic_distances 
                                          if d == min_phonetic_distance and w not in used]
            else:
                replacement_candidates = [w for w, d in candidate_phonetic_distances if w not in used]             
                
            # из кандидатов берем максимально близкое теме слово
            embeddings_distances = [
                (replacement_word, emb.distance(seed_vec, emb.word_vector(replacement_word)))
                for replacement_word in replacement_candidates
                ]
            embeddings_distances.sort(key=lambda pair: pair[1])
            if not embeddings_distances:
#                 print('no near emb', word)
                continue
            embeddings_nearest = [k for k,v in embeddings_distances[:3]]
            new_word = np.random.choice(embeddings_nearest)
            
            if poem[li][ti] != new_word:
                poem[li][ti] = new_word
                replaced += 1
                used.add(new_word)

    # собираем получившееся стихотворение из слов
    generated_poem = '\n'.join([' '.join([token for token in line]).capitalize() for line in poem])
    clean = generated_poem.replace('« ', '«').replace(' »', '»').replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?').replace(' :', ':').replace(' ;', ';')
    print('%.1f' % (100*replaced/total))
    return clean
        
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
    pids = ['pushkin', 'esenin', 'mayakovskij', 'blok', 'tyutchev']
    print('run server')
    with open('token.txt') as f:
        token = f.read().strip()
    base_url = 'https://api.telegram.org/bot%s/' % token
    print('base_url', base_url)
    start_update = requests.get(base_url + 'getUpdates').json()['result']
    last_id = start_update[-1]['update_id'] if start_update else 0
    print('ready')
    while True:
        updates = start_update = requests.get(base_url + 'getUpdates?offset=%d' % (last_id+1)).json()['result']
        for u in updates:
            print('upd', u)
            last_id = u['update_id']
            seed = u['message']['text']
            uid = u['message']['chat']['id']
            poet_id = np.random.choice(pids)
            poem = generate_poem(seed, poet_id)
            print(poet_id, ':', poem)
            send = base_url + 'sendMessage?chat_id=%d&text=%s' % (uid, poet_id+'\n'+poem)
            requests.get(send)
            print('sent')
    # app.config['JSON_AS_ASCII'] = False
    # app.run(host='0.0.0.0', port=8000)