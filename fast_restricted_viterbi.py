import numpy as np


"""
FastRestrictedHMM: Perform the Viterbi Algorithm for HTMM inference

@Parameters:
- t: int
- topics: int
- states: int

@Methods:
- viterbi(epsilon, theta, local, pi, best_path): the algorithm
"""

class FastRestrictedViterbi:
    def __init__(self):
        self.t_ = self.topics_ = self.states_ = 0


    def viterbi(self, epsilon, theta, local, pi, best_path):
        self.t_ = len(local)
        self.topics_ = len(theta)
        self.states_ = self.topics_ * 2
        delta = np.zeros((self.t_, self.states_))
        best = np.zeros((self.t_, self.states_), dtype='int')
        self.compute_all_deltas(pi, local, theta, epsilon, delta, best)
        self.backtrack_best_path(delta[-1], best, best_path)


    def compute_all_deltas(self, pi, local, theta, epsilon, delta, best):
        self.init_delta(pi, local[0], delta[0])
        for i in range(1, self.t_):
            prev_best = np.argmax(delta[i-1]) # this is find_best_in_level
            for s in range(self.topics_):
                delta[i, s] = delta[i-1, prev_best] * epsilon * theta[s] * local[i, s]
                best[i, s] = prev_best

                if delta[i-1, s] > delta[i-1, s+self.topics_]:
                    delta[i, s+self.topics_] = delta[i-1, s] * (1-epsilon) * local[i, s]
                    best[i, s+self.topics_] = s
                else:
                    delta[i, s+self.topics_] = delta[i-1, s+self.topics_] * (1-epsilon) * local[i, s]
                    best[i, s+self.topics_] = s + self.topics_

            delta[i] /= delta[i].sum()


    def init_delta(self, pi, local0, delta0):
        for i in range(self.topics_):
            delta0[i] = pi[i] * local0[i]
            delta0[i+self.topics_] = pi[i+self.topics_] * local0[i]
        delta0 /= delta0.sum()


    def backtrack_best_path(self, delta_t, best, best_path):
        best_path[-1] = np.argmax(delta_t)
        for i in range(self.t_-2, -1, -1):
            best_path[i] = best[i+1, best_path[i+1]]
