import json
from utils import *
word_index = {} #string: int
index_word = {} #int: string
index = 0
def process_json(json_filename):
    product = json.load(open(json_filename))
    docs = []
    for review in product['Reviews']:
        docs.append(process(review['Content']))
    return docs

def process(txt):
    sentences = paragraph2sentence(txt)
    doc = _Document()
    for stn in sentences:
        sentence = _Sentence()
        for w in sentence2word_normalized(stn):
            if w in word_index:
                sentence.add_word(word_index[w])
            else:
                word_index[w] = index
                index_word[index] = w
                index += 1

        doc.add_sentence(sentence)
    return doc
