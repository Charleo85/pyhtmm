import random, math, pickle, sys
import numpy as np
from tqdm import tqdm

from multiprocessing import Process, Queue
from multiprocessing.sharedctypes import RawArray, Array

from .fast_restricted_hmm import FastRestrictedHMM
from .fast_restricted_viterbi import FastRestrictedViterbi
from .utils import word2index

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
HTMM: the Hidden Topic Markov Model

@Parameters:
- topics: int
- words: int
- alpha: float
- beta: float
- epsilon: float
- iters: int
- phi: numpy.ndarray(topics, words)

@Methods:
- predict_topic(doc): predict topic for each sentence in a document
"""
class HTMM(Pickleable):
    def __init__(self, alpha, beta, epsilon, iters, phi, topics, words):
        self.alpha_ = alpha
        self.beta_ = beta
        self.epsilon_ = epsilon
        self.iters_ = iters
        self.phi_ = phi
        self.topics_ = topics
        self.words_ = words


    def predict_topic(self, doc):
        theta_i = np.random.rand(self.topics_)
        p_dwzpsi_i = np.zeros((doc.num_sentences, 2*self.topics_))
        path = [0] * doc.num_sentences

        for i in range(self.iters_):
            self.e_step_in_single_doc_htmm(doc, theta_i, p_dwzpsi_i)
            self.find_theta_htmm(doc, theta_i, p_dwzpsi_i)

        local = np.zeros((doc.num_sentences, self.topics_))
        self.compute_local_probs_for_doc_htmm(doc, local)
        init_probs = np.zeros(2*self.topics_)
        for i in range(self.topics_):
            init_probs[i] = theta_i[i]
            init_probs[i+self.topics_] = 0.0

        f = FastRestrictedViterbi()
        f.viterbi(self.epsilon_, theta_i, local, init_probs, path)

        return path


    def e_step_in_single_doc_htmm(self, doc, theta_i, p_dwzpsi_i):
        local = np.zeros((doc.num_sentences, self.topics_))
        self.compute_local_probs_for_doc_htmm(doc, local)

        init_probs = np.zeros(2*self.topics_)
        for i in range(self.topics_):
            init_probs[i] = theta_i[i]
            init_probs[i + self.topics_] = 0.0

        f = FastRestrictedHMM()
        f.forward_backward(self.epsilon_, theta_i, local, init_probs, p_dwzpsi_i)


    def compute_local_probs_for_doc_htmm(self, doc, local):
        for i in range(doc.num_sentences):
            for z in range(self.topics_):
                local[i, z] = 1.0 / self.topics_

            print(doc.sentence_list[i].word_list)
            for j in range(doc.sentence_list[i].num_words):
                norm = 0.0
                word = doc.sentence_list[i].word_list[j]
                for z in range(self.topics_):
                    local[i, z] *= self.phi_[z][word]
                    norm += local[i, z]
                local[i] /= norm


    def find_theta_htmm(self, doc, theta_i, p_dwzpsi_i):
        cdz = np.zeros(self.topics_)
        for i in range(doc.num_sentences):
            for z in range(self.topics_):
                cdz[z] += p_dwzpsi_i[i, z]

        for z in range(self.topics_):
            theta_i[z] = cdz[z] + self.alpha_ - 1
        theta_i /= theta_i.sum()


"""
EM: A EM training wrapper of the Hidden Topic Markov Model

@Parameters:
- topics: int
- words: int
- alpha: float
- beta: float
- docs: list<Document>
- epsilon: float
- iters: int
- theta: numpy.ndarray(len(docs), topics)
- phi: numpy.ndarray(topics, words)
- p_dwzpsi: numpy.ndarray(len(docs), "# sentences in a doc", 2*topics)
- p_dwzpsi_shape: numpy.ndarray.shape
- loglik: float

@Methods:
- infer(): train the model on the set of training documents (docs)
- map_topic_estimate(idx, path): inference on doc #idx and set "path"
- print_top_word(index_word, K): print K top words in each topic
- load_prior(file, word_index, eta): load prior probs into the model
"""
class EM(HTMM):
    def __init__(self, doc, words, topics=10, alpha=1.001, beta=1.0001, iters=100, num_workers=1):
        super(EM, self).__init__(alpha, beta, 0, iters, None, topics, words)
        self.docs_ = doc
        self.num_workers_ = num_workers
        self.rand_init_params()
        self.loglik_ = 0.0


    def save_HTMM_model(self, filepath):
        htmm = HTMM(self.alpha_, self.beta_, self.epsilon_, self.iters_, self.phi_, self.topics_, self.words_)
        htmm.save(filepath)


    def infer(self, iters=None):
        if iters is None: iters = self.iters_
        shared_arr = None
        if self.num_workers_ > 1:
            shared_arr = RawArray('d', self.p_dwzpsi_.flatten())
            tmp = np.frombuffer(shared_arr)
            self.p_dwzpsi_ = tmp.reshape(self.p_dwzpsi_shape_)

        for epoch in tqdm(range(iters)):
            self.e_step(shared_arr)
            self.m_step()
            tqdm.write("iteration: %d, loglikelihood: %f" % (epoch, self.loglik_))

        if self.num_workers_ > 1:
            self.p_dwzpsi_ = np.copy(self.p_dwzpsi_)


    def map_topic_estimate(self, idx):
        path = [0] * self.docs_[idx].num_sentences

        local = np.zeros(self.docs_[idx].num_sentences, self.topics_)
        self.compute_local_probs_for_doc(self.docs_[idx], local)
        init_probs = np.zeros(self.topics_*2)
        for i in range(self.topics_):
            init_probs[i] = self.theta_[idx][i]
            init_probs[i+self.topics_] = 0.0

        f = FastRestrictedViterbi()
        f.viterbi(self.epsilon_, self.theta_[idx], local, init_probs, path)

        return path


    def print_top_word(self, index_word, K=10):
        for phi in self.phi_:
            for idx in np.argsort(phi)[-K:]:
                print(index_word[idx])
            print("=" * 10)


    def load_prior(self, prior_file, word_index, eta=5.0):
        lines = open(prior_file, 'r')
        for z, l in enumerate(lines):
            for raw_word in l.rstrip('\n').split(' ')[1:]:
                word = word2index(raw_word)
                if word in word_index:
                    idx = word_index[word]
                    self.phi_[z, idx] += eta
        lines.close()

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

        init_probs = np.zeros(2*self.topics_)
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
                local[i, z] = 1.0 / self.topics_

            ret += math.log(self.topics_)
            for j in range(doc.sentence_list[i].num_words):
                norm = 0.0
                word = doc.sentence_list[i].word_list[j]
                for z in range(self.topics_):
                    local[i, z] *= self.phi_[z][word]
                    norm += local[i, z]
                local[i] /= norm
                if norm <= 0.0: continue
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
