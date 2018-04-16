import random, math, pickle, queue
import numpy as np

from multiprocessing import Process, Queue
from multiprocessing.sharedctypes import RawArray, Array

from fast_restricted_hmm import FastRestrictedHMM
from fast_restricted_viterbi import FastRestrictedViterbi
from process import read_train_documents
from utils import config_logger, save_pickle, load_pickle

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
- p_dwzpsi: numpy.ndarray(len(docs), # sentences in a doc", 2*topics)
- loglik: float

@Methods:
- inter(): train the model on the set of training documents (docs)
- map_topic_estimate(idx, path): inference on doc #idx and set "path"
"""

class HTMM(Pickleable):
    def __init__(self, doc, words, topics=10, alpha=1.001, beta=1.0001, iters=100, num_workers=10):
        self.topics_ = topics
        self.words_ = words
        self.alpha_ = alpha
        self.beta_ = beta
        self.iters_ = iters
        self.docs_ = doc
        self.num_workers_ = num_workers
        self.rand_init_params()
        self.loglik_ = 0.0


    def infer(self):
        shared_arr = None
        if self.num_workers_ > 1:
            shared_arr = RawArray('d', self.p_dwzpsi_.flatten())
            tmp = np.frombuffer(shared_arr)
            self.p_dwzpsi_ = tmp.reshape(self.p_dwzpsi_shape_)

        for epoch in range(self.iters_):
            self.e_step(shared_arr)
            self.m_step()
            print("iteration: %d, loglikelihood: %f" % (epoch, self.loglik_))

        if self.num_workers_ > 1:
            self.p_dwzpsi_ = np.copy(self.p_dwzpsi_)


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

        max_dim = 0
        for i in range(len(self.docs_)):
            max_dim = max(max_dim, self.docs_[i].num_sentences)
        self.p_dwzpsi_ = np.zeros((len(self.docs_), max_dim, 2*self.topics_))
        self.p_dwzpsi_shape_ = self.p_dwzpsi_.shape


    def e_step(self, shared_arr):
        # print("before e step")
        assert(self.num_workers_ > 1 or shared_arr is None)

        self.loglik_ = 0.0
        if self.num_workers_ > 1:
            q = Queue()
            ps = [Process(target=self.e_step_chunk, args=(i, q, shared_arr)) for i in range(self.num_workers_)]
            for p in ps: p.start()
            for p in ps: p.join()
            while not q.empty(): self.loglik_ += q.get()
        else:
            for d in range(len(self.docs_)):
                self.loglik_ += self.e_step_in_single_doc(d, self.p_dwzpsi_)

        self.interpret_priors_into_likelihood()


    def e_step_chunk(self, pid, q, shared_arr):
        chunk_len = len(self.docs_) // (self.num_workers_ - 1)
        start_idx = pid * chunk_len
        end_idx = min(len(self.docs_), (pid+1) * chunk_len)

        tmp = np.frombuffer(shared_arr)
        p_dwzpsi_ptr = tmp.reshape(self.p_dwzpsi_shape_)

        ret = 0.0
        for d in range(start_idx, end_idx):
            ret += self.e_step_in_single_doc(d, p_dwzpsi_ptr)
        q.put(ret)


    def e_step_in_single_doc(self, idx, p_dwzpsi_ptr):
        ret = 0.0
        doc = self.docs_[idx]

        local = np.zeros((doc.num_sentences, self.topics_))
        ret += self.compute_local_probs_for_doc(doc, local)

        init_probs = np.zeros(self.topics_ * 2)
        for i in range(self.topics_):
            init_probs[i] = self.theta_[idx][i]
            init_probs[i + self.topics_] = 0.0

        f = FastRestrictedHMM()
        ret += f.forward_backward(self.epsilon_, self.theta_[idx], local, init_probs, p_dwzpsi_ptr[idx])

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
        # print("before m step")
        self.find_epsilon()
        # print("after epsilon")
        self.find_phi()
        # print("after phi")
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


    def find_phi(self):
        czw = self.count_topic_words()
        for z in range(self.topics_):
            for w in range(self.words_):
                self.phi_[z, w] = czw[z, w] + self.beta_ - 1
            self.phi_[z] /= self.phi_[z].sum()


    def count_topic_words(self):
        czw = np.zeros((self.topics_, self.words_))

        if self.num_workers_ > 1:
            shared_arr = Array('d', czw.flatten())
            ps = [Process(target=self.czw_chunk, args=(i, shared_arr)) for i in range(self.num_workers_)]
            for p in ps: p.start()
            for p in ps: p.join()
            tmp = np.frombuffer(shared_arr.get_obj())
            czw = np.copy(tmp.reshape((self.topics_, self.words_)))
        else:
            for d in range(len(self.docs_)):
                for i in range(self.docs_[d].num_sentences):
                    sen = self.docs_[d].sentence_list[i]
                    for w in sen.word_list:
                        for z in range(self.topics_):
                            czw[z, w] += self.p_dwzpsi_[d][i][z] + self.p_dwzpsi_[d][i][z+self.topics_]

        return czw


    def czw_chunk(self, pid, shared_arr):
        chunk_len = len(self.docs_) // (self.num_workers_ - 1)
        start_idx = pid * chunk_len
        end_idx = min(len(self.docs_), (pid+1) * chunk_len)

        tmp = np.frombuffer(shared_arr.get_obj())
        czw = tmp.reshape((self.topics_, self.words_))

        for d in range(start_idx, end_idx):
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
            for idx in np.argsort(phi)[:K]:
                print(index_word[idx])
            print("=" * 10)


    def load_prior(self, prior_file, eta=5.0):
        pass


if __name__ == "__main__":
    word_index_filepath = './data/pickle/word_index.pickle'
    index_word_filepath = './data/pickle/index_word.pickle'
    model_filepath = './data/pickle/model.pickle'
    docs_path = './data/pickle/docs.pickle'

    # config_logger()
    try:
        word_index = load_pickle(word_index_filepath)
        index_word = load_pickle(index_word_filepath)
        docs = load_pickle(docs_path)
    except:
        docs, word_index, index_word = read_train_documents('./data/laptops/')
        save_pickle(word_index, word_index_filepath)
        save_pickle(index_word, index_word_filepath)
        save_pickle(docs, docs_path)

    try:
        model = load_pickle(model_filepath)
    except:
        model = HTMM(docs, len(word_index), iters=100)
        model.save(model_filepath)

    model.load_prior('laptops_bootstrapping_test.dat')
    model.infer()
    model.print_top_word(index_word, 15)
