from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from nltk import tokenize
# from nltk.tokenize.treebank import TreebankWordTokenizer
# from nltk.tokenize.stanford import StanfordTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.snowball import SnowballStemmer
import re, pickle
import locale

sid = SentimentIntensityAnalyzer()
# word_tokenize = TreebankWordTokenizer().tokenize
# word_tokenize = CoreNLPParser(options={"americanize": True, "ptb3Escaping": True, "splitHyphenated": False}).tokenize
ss = SnowballStemmer('english')
url_regex = re.compile('(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?')

def clean_up(raw_sentence):
    out = re.sub(url_regex, 'URLADDR', raw_sentence)
    out = out.replace('/', ' / ')
    # out = out.replace('*', ' * ')
    out = out.replace('…', ' … ')
    out = out.replace('=', ' = ')
    out = out.replace('+', ' + ')
    out = out.replace('.', '')
    out = out.replace(',', '')
    out = out.replace( u'\u200b', ' ')
    out = out.replace( u'\u200e', '')
    out = out.replace( u'\u2018', u"'")
    out = out.replace( u'\u2018', u"'")
    out = out.replace( u'\u2019', u"'")
    out = out.replace( u'\u201c', u'"')
    out = out.replace( u'\u201d', u'"')
    return out

def normalized(word):
    w = word.lower()
    while len(w) > 1 and w.startswith('-'): w = w[1:]
    while len(w) > 1 and w.endswith('-'): w = w[:-1]
    w = ss.stem(w)
    if w.isdigit(): return "NUM"
    else: return w

def paragraph2sentence(para):
    return sent_tokenize(para)

def sentence2word(raw_sentence):
    cleaned_sentence = clean_up(raw_sentence)
    return word_tokenize(cleaned_sentence)

def sentence2word_normalized(raw_sentence):
    return [normalized(w) for w in sentence2word(raw_sentence)]

def sentiment(raw_sentence):
    return id.polarity_scores(raw_sentence.lower())['compound']

def numSentence(para):
    return sum(1 for sent in paragraph2sentence(para) if sent.length > 0)

def save_pickle(data, filename='sample.pickle'):
	with open(filename, mode='wb') as f:
		pickle.dump(data, f)

def save_txt(data, filename='sample.txt'):
	with open(filename, mode='w') as f:
		f.write(str(data))
