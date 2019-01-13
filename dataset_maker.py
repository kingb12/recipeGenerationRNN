import os
import sys
import json
import random
import nltk

PAD = '<pad>'
MASK = '<mask>'
BEGIN = '<beg>'
END = '<end>'
BEGIN_INGREDIENTS = '<begin_ingredients>'
ING = '<ing>'
STEP = '<step>'
TOKEN_STEP = '___step___'
TOKEN_ING = '___ing___'

DEFAULT_CONFIG = {'max_in_length': 200,
                  'max_out_length': 300,
                  'section_size': 10000,
                  'include_ingredients': True,
                  'prune_ingredients': False,
                  'split_ingredients': False,
                  'max_ingredients': 20,
                  'stop_words': 'stop_words.json',
                  'title_last': False,
                  'data_dir': '.',
                  'file_prefix': 'rl_',
                  'noise_tail': 0}
config = DEFAULT_CONFIG
if len(sys.argv) > 1:
    with open(sys.argv[1], 'r') as f:
        config.update(json.loads(f.read()))

all_recipes = os.listdir(config['data_dir'])
#Shuffle them for randomly assorted batches
random.shuffle(all_recipes)

#important pad is the first word
word_to_n = {'<mask>': 0}
word_freq = {0: 0}

start = 0
end = config['section_size']

def process_sequence(sequence, word_to_n, word_freq):
    """
    Generates a list of numbers from list of words or list of list of numbers from list of list of words
    """
    if type(sequence[0]) == list:
        return [process_sequence(s, word_to_n, word_freq) for s in sequence]
    result = []
    for w in sequence:
        if w in word_to_n:
            word_freq[word_to_n[w]] += 1
        else:
            word_to_n[w] = len(word_to_n)
            word_freq[word_to_n[w]] = 1
        result.append(word_to_n[w])
    return result

with open(config['stop_words'], 'r') as f:
    stop_words = set(json.loads(f.read()))

def prune_ingredients(ingredients, stop_words):
    if type(ingredients[0]) == list:
        return [prune_ingredients(ings, stop_words) for ings in ingredients]
    return [ing for ing in ingredients if ing not in stop_words]

def sequential_inputs(rec, cfg):
    title_ingredients = nltk.word_tokenize(rec['name'])
    max_ing = config['max_ingredients']
    if cfg['include_ingredients']:
        ingredients = nltk.word_tokenize((" " + TOKEN_ING + " ").join(rec['ingredientLines'][0:max_ing]))
    if cfg['prune_ingredients']:
        ingredients = prune_ingredients(ingredients, stop_words)
        if cfg['title_last']:
            title_ingredients.insert(0, BEGIN_INGREDIENTS)
            title_ingredients = ingredients + title_ingredients
        else:
            title_ingredients.append(BEGIN_INGREDIENTS)
            title_ingredients.extend(ingredients)
        title_ingredients = [ti if ti != TOKEN_ING else ING for ti in title_ingredients]
    if len(title_ingredients) > cfg['max_in_length']:
        title_ingredients = title_ingredients[0:cfg['max_in_length']]
    in_length = len(title_ingredients)
    pad = [PAD for i in range(len(title_ingredients), cfg['max_in_length'])]
    pad.extend(title_ingredients)
    ei = pad
    return process_sequence(ei, word_to_n, word_freq), in_length

def split_inputs(rec, cfg):
    title = nltk.word_tokenize(rec['name'])
    max_ing = config['max_ingredients']
    if cfg['include_ingredients']:
        ingredients = [nltk.word_tokenize(r) for r in rec['ingredientLines'][0:max_ing]]
        pad = [[PAD] * config['max_in_length'] for i in range(len(ingredients), cfg['max_ingredients'])]
        ingredients.extend(pad)
        if cfg['prune_ingredients']:
            ingredients = prune_ingredients(ingredients, stop_words)
        ingredients.insert(0, title)
        title_ingredients = ingredients
        for seq in title_ingredients:
            seq = [ti if ti != TOKEN_ING else ING for ti in seq]
    else:
        title_ingredients = [title]
    for j, seq in enumerate(title_ingredients):
        if len(seq) > cfg['max_in_length']:
            title_ingredients[j] = seq[0:cfg['max_in_length']]
    in_length = sum([len(seq) for seq in title_ingredients])
    for j, seq in enumerate(title_ingredients):
        pad = [PAD for i in range(len(seq), cfg['max_in_length'])]
        pad.extend(seq)
        title_ingredients[j] = pad
    ei = title_ingredients
    return process_sequence(ei, word_to_n, word_freq), in_length

altnum = 0
while end <= len(all_recipes):
    recipes = all_recipes[start:end]
    section_num = start / config['section_size']
    start += config['section_size']
    end += config['section_size']

    # Go through each file in data directory and load the recipe as a JSON
    # needed fields: name, ingredientLines, preparationSteps
    enc_inputs = []
    in_lengths = []
    dec_inputs = []
    out_lengths = []
    outputs = []
    for fname in recipes:
        with open(config['data_dir'] + '/' + fname, 'r') as f:
            recipe = json.loads(f.read())
        # create 3 vectors for each recipe:
        # Enc_Input: <pad> .... title words <begin_ingredients> ingredients (<ing> separated) <end>
        if len(recipe['name']) == 0 or len(recipe['ingredientLines']) == 0 or len(recipe['preparationSteps']) == 0 or recipe['preparationSteps'][0] is None:
            recipes.append(all_recipes[altnum])
            altnum -= 1
            continue
        if config['split_ingredients']:
            enc_input, in_length = split_inputs(recipe, config)
        else:
            enc_input, in_length = sequential_inputs(recipe, config)
        # Dec_Input: <beg> prep steps . <pad>
        steps = nltk.word_tokenize((' ' + TOKEN_STEP + ' ').join(recipe['preparationSteps']))
        steps = [s if s != TOKEN_STEP else STEP for s in steps]
        steps.insert(0, BEGIN)
        if len(steps) > config['max_out_length']:
            steps = steps[0:config['max_out_length']]
        noise = 0
        if config['noise_tail'] > 0:
            noise = random.randint(0, config['noise_tail'])
            steps[(-1*noise):] = [PAD] * noise
        out_length = len(steps)
        padding = [PAD for i in range(len(steps), config['max_out_length'])]
        steps.extend(padding)
        dec_input = steps
        # Output: prep steps . <end> <pad>
        steps = nltk.word_tokenize((' ' + TOKEN_STEP + ' ').join(recipe['preparationSteps']))
        steps = [s if s != TOKEN_STEP else STEP for s in steps]
        steps.append(END)
        if len(steps) > config['max_out_length']:
            steps = steps[0:config['max_out_length']]
        if config['noise_tail'] > 0:
            steps[(-1*noise):] = [PAD] * noise
        masking = [MASK for i in range(len(steps), config['max_out_length'])]
        steps.extend(masking)
        output = steps
        # NOTE THE SIDE EFFECTS: Only do this once per input in order to maintain accurate counts
        enc_inputs.append(enc_input)
        in_lengths.append(in_length)
        dec_inputs.append(process_sequence(dec_input, word_to_n, word_freq))
        out_lengths.append(out_length)
        outputs.append(process_sequence(output, word_to_n, word_freq))
    with open(config['file_prefix'] + 'enc_inputs_' + str(section_num) +  '.json', 'w') as f:
        f.write(json.dumps(enc_inputs))
    with open(config['file_prefix'] + 'dec_inputs_' + str(section_num) +  '.json', 'w') as f:
        f.write(json.dumps(dec_inputs))
    with open(config['file_prefix'] + 'outputs_' + str(section_num) +  '.json', 'w') as f:
        f.write(json.dumps(outputs))
    with open(config['file_prefix'] + 'in_lengths_' + str(section_num) +  '.json', 'w') as f:
            f.write(json.dumps(in_lengths))
    with open(config['file_prefix'] + 'out_lengths_' + str(section_num) +  '.json', 'w') as f:
        f.write(json.dumps(out_lengths))
helper = {'w_to_n': word_to_n, 'n_to_w': dict([(v, k) for (k, v) in word_to_n.items()]), 'weights': word_freq}
with open(config['file_prefix'] + 'helper.json', 'w') as f:
    f.write(json.dumps(helper))


