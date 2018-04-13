import random, math, pickle
import numpy as np

from fast_restricted_hmm import FastRestrictedHMM
from fast_restricted_viterbi import FastRestrictedViterbi
from process import read_train_documents
from utils import config_logger, save_pickle, load_pickle, word2index
from sys import argv

"""
Pickleable: An interface for loading and saving objects with pickle

@Methods:
- load(filepath): load this object from a file named "filepath"
- save(filepath): save current state of object to file "filepath"
"""

class Pickleable:
    def save(self, filepath):
        data_file = open(filepath, 'wb')
        pickle.dump(self, data_file)
        data_file.close()

    def load(self, filepath):
        data_file = open(filepath, 'rb')
        obj = pickle.load(data_file)
        for attr, value in obj.__dict__.items():
            setattr(self, attr, value)
        data_file.close()


"""
HTMM: A Python implementation of the Hidden Topic Markov Model

@Parameters:
- topics: int
- words: int
- alpha: float
- beta: float
- docs: list<Document>
- epsilon: float
- theta: numpy.ndarray(len(docs), topics)
- phi: numpy.ndarray(topics, words)
- p_dwzpsi: list<numpy.ndarray(""# sentences in a doc", 2*topics)>
- loglik: float

@Methods:
- inter(): train the model on the set of training documents (docs)
- map_topic_estimate(idx, path): inference on doc #idx and set "path"
"""

class HTMM(Pickleable):
    def __init__(self, doc, words, topics=10, alpha=1.001, beta=1.0001):
        self.topics_ = topics
        self.words_ = words
        self.alpha_ = alpha
        self.beta_ = beta
        self.docs_ = doc
        self.rand_init_params()
        self.loglik_ = 0.0


    def infer(self, iters=100):
        for epoch in range(iters):
            self.e_step()
            self.m_step()
            print("iteration: %d, loglikelihood: %f" % (epoch, self.loglik_))


    def map_topic_estimate(self, idx, path):
        f = FastRestrictedViterbi()
        local = np.zeros(self.docs_[idx].num_sentences, self.topics_)
        self.compute_local_probs_for_doc(self.docs_[idx], local)
        init_probs = np.zeros(self.topics_*2)
        for i in range(self.topics_):
            init_probs[i] = self.theta_[idx][i]
            init_probs[i+self.topics_] = 0.0
        f.viterbi(self.epsilon_, self.theta_[idx], local, init_probs, path)


    def rand_init_params(self):
        self.epsilon_ = random.uniform(0, 1)

        self.theta_ = np.random.rand(len(self.docs_), self.topics_)
        for i in range(self.theta_.shape[0]):
            self.theta_[i] /= self.theta_[i].sum()

        self.phi_ = np.random.rand(self.topics_, self.words_)
        for i in range(self.phi_.shape[0]):
            self.phi_[i] /= self.phi_[i].sum()

        self.p_dwzpsi_ = [None] * len(self.docs_)
        for i in range(len(self.p_dwzpsi_)):
            self.p_dwzpsi_[i] = np.zeros((self.docs_[i].num_sentences, 2*self.topics_))


    def e_step(self):
        self.loglik_ = 0.0
        for i in range(len(self.docs_)):
            self.loglik_ += self.e_step_in_single_doc(i)
        self.interpret_priors_into_likelihood()


    def e_step_in_single_doc(self, idx):
        ret = 0.0
        doc = self.docs_[idx]

        local = np.zeros((doc.num_sentences, self.topics_))
        ret += self.compute_local_probs_for_doc(doc, local)

        init_probs = np.zeros(self.topics_ * 2)
        for i in range(self.topics_):
            init_probs[i] = self.theta_[idx][i]
            init_probs[i + self.topics_] = 0.0

        f = FastRestrictedHMM()
        ret += f.forward_backward(self.epsilon_, self.theta_[idx], local, init_probs, self.p_dwzpsi_[idx])

        return ret


    def compute_local_probs_for_doc(self, doc, local):
        ret = 0.0

        for i in range(doc.num_sentences):
            for z in range(self.topics_):
                local[i, z] = 1.0 / float(self.topics_)

            ret += math.log(self.topics_)
            for j in range(doc.sentence_list[i].num_words):
                norm = 0.0
                word = doc.sentence_list[i].word_list[j]
                for z in range(self.topics_):
                    local[i, z] *= self.phi_[z][word]
                    norm += local[i, z]
                local[i] /= norm
                if norm <= 0: continue
                ret += math.log(norm)

        return ret


    def interpret_priors_into_likelihood(self):
        for d in range(len(self.docs_)):
            for z in range(self.topics_):
                if self.theta_[d][z] <= 0: continue
                self.loglik_ += (self.alpha_ - 1) * math.log(self.theta_[d][z])

        for z in range(self.topics_):
            for w in range(self.words_):
                if self.phi_[z][w] <= 0: continue
                self.loglik_ += (self.beta_ - 1) * math.log(self.phi_[z][w])


    def m_step(self):
        self.find_epsilon()
        self.find_phi()
        self.find_theta()


    def find_epsilon(self):
        total = 0
        lot = 0.0
        for d in range(len(self.docs_)):
            for i in range(1, self.docs_[d].num_sentences):
                for z in range(self.topics_):
                    lot += self.p_dwzpsi_[d][i][z]
            total += self.docs_[d].num_sentences - 1
        self.epsilon_ = lot / total
        print(self.epsilon_)

    def find_phi(self):
        czw = np.zeros((self.topics_, self.words_))
        self.count_topic_words(czw)

        for z in range(self.topics_):
            for w in range(self.words_):
                self.phi_[z, w] = czw[z, w] + self.beta_ - 1
            self.phi_[z] /= self.phi_[z].sum()


    def count_topic_words(self, czw):
        for d in range(len(self.docs_)):
            for i in range(self.docs_[d].num_sentences):
                sen = self.docs_[d].sentence_list[i]

                for w in sen.word_list:
                    for z in range(self.topics_):
                        czw[z, w] += self.p_dwzpsi_[d][i][z] + self.p_dwzpsi_[d][i][z+self.topics_]


    def find_theta(self):
        for d in range(len(self.docs_)):
            cdz = np.zeros(self.topics_)
            for i in range(self.docs_[d].num_sentences):
                for z in range(self.topics_):
                    cdz[z] += self.p_dwzpsi_[d][i][z]

            for z in range(self.topics_):
                self.theta_[d, z] = cdz[z] + self.alpha_ - 1
            self.theta_[d] /= self.theta_[d].sum()

    def print_top_word(self, index_word, K=10):
        for phi in self.phi_:
            for idx in np.argsort(phi)[-K:]:
                print(index_word[idx])
            print("==========")

    def load_prior(self, prior_file, word_index, eta=5.0):
        with open(prior_file, 'r') as lines:
            for i, l in enumerate(lines):
                for raw_word in l.rstrip('\n').split(' ')[1:]:
                    word = word2index(raw_word)
                    if word in word_index:
                        idx = word_index[word]
                        self.phi_[i, idx] += eta






word_index_filepath = './data/pickle/word_index.pickle'
index_word_filepath = './data/pickle/index_word.pickle'
model_filepath = './data/pickle/model.pickle'
model_trained_filepath = './data/pickle/trained_model.pickle'

if __name__ == "__main__":
    # config_logger()
    try:
        word_index = load_pickle(word_index_filepath)
        index_word = load_pickle(index_word_filepath)
        num_words = len(index_word)
    except:
        docs, num_words, word_index, index_word = read_train_documents('./data/laptops/') #use ./data/debug/ for debugging
        save_pickle(word_index, word_index_filepath)
        save_pickle(index_word, index_word_filepath)

    print(argv)
    if argv[1] == 'infer':
        ### print topword in trained model
        model = load_pickle(model_trained_filepath)
        model.print_top_word(index_word, 25)
    else:
        ## train model
        try:
            model = load_pickle(model_filepath)
        except:
            model = HTMM(docs, num_words)
            model.save(model_filepath)

        # print(num_words, word_index)
        model.load_prior('./data/laptops_bootstrapping_test.dat', word_index)
        model.infer(iters=5)
        model.print_top_word(index_word, 15)
        model.save(model_trained_filepath)
