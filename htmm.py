import random, math, pickle
import numpy as np

from fast_restricted_hmm import FastRestrictedHMM
from process import process_json

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
- iters: int
- docs: list<Document>
- epsilon: float
- theta: numpy.ndarray(len(docs), topics)
- phi: numpy.ndarray(topics, words)
- p_dwzpsi: list<numpy.ndarray(""# sentences in a doc", 2*topics)>
- loglik: float

@Methods:
- inter(): train the model on the set of training documents (docs)
"""

class HTMM(Pickleable):
    def __init__(self, topics, words, alpha, beta, iters, fname, data_dir):
        self.topics_ = topics
        self.words_ = words
        self.alpha_ = alpha
        self.beta_ = beta
        self.iters_ = iters
        self.docs_ = []
        self.read_train_documents(fname, data_dir)
        self.rand_init_params()
        self.loglik_ = 0.0


    def read_train_documents(self, data_dir):
        for filename in os.listdir(data_dir):
            self.docs_ += process(data_dir+filename)


    def rand_init_params(self):
        self.epsilon_ = random.uniform(0, 1)

        self.theta_ = np.random.rand(len(self.docs_), self.topics_)
        for i in range(self.theta_.shape[0]):
            self.theta_[i] /= self.theta_[i].sum()

        self.phi_ = np.random.rand(self.topics_, self.words_)
        for i in range(self.phi_.shape[0]):
            self.phi_[i] /= self.phi_[i].sum()

        self.p_dwzpsi_ = [None] * len(self.docs_)
        for i in range(self.p_dwzpsi_):
            self.p_dwzpsi_[i] = np.random.rand(self.docs_[i].num_sentences, 2*self.topics_)


    def infer(self):
        for epoch in range(self.iters_):
            self.e_step()
            self.m_step()
            print("iteration: %d, loglikelihood: %f" % (epoch, self.loglik_))


    def e_step(self):
        self.loglik_ = 0.0
        for i in range(len(self.docs_)):
            self.loglik_ += self.e_step_in_single_doc(i)


    def e_step_in_single_doc(self, idx):
        ret = 0.0
        doc = self.docs_[idx]

        local = np.zeros(doc.num_sentences, self.topics_)
        ret += self.compute_local_probs_for_doc(doc, local)

        init_probs = np.zeros(self.topics_ * 2)
        for i in range(self.topics_):
            init_probs[i] = self.theta_[idx][i]
            init_probs[i + self.topics_] = 0.0

        f = FastRestrictedHMM()
        ret += f.forward_backward(self.epsilon_, theta_[idx], local, init_probs, self.p_dwzpsi_[idx])

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
                ret += math.log(norm)

        return ret


    def m_step(self):
        self.find_epsilon()
        self.find_phi()
        self.find_theta()


    def find_epsilon(self):
        total = 0
        lot = 0.0
        for d in range(len(self.docs_)):
            for i in range(self.docs_[d].num_sentences):
                for z in range(self.topics_):
                    lot += self.p_dwzpsi_[d][i][z]
            total += self.docs_[d].num_sentences - 1
        self.epsilon_ = lot / float(total)


    def find_phi(self):
        czw = np.zeros(self.topics_, self.words_)
        self.count_topic_words(czw)

        for z in range(self.topics_):
            for w in range(self.words_):
                self.phi_[z, w] = czw[z, w] + self.beta_ - 1
            self.phi_[z] /= self.phi_[z].sum()


    def count_topic_words(self, czw):
        for d in range(len(self.docs_)):
            for i in range(self.docs_[d].num_sentences):
                sen = self.docs_[d].sentence_list[i]

                for word in sen.word_list:
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
