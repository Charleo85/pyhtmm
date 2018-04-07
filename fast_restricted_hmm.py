import numpy as np


"""
FastRestrictedHMM: Provide computation in a Hidden Markov Model

@Parameters:
- topics: int
- states: int

@Methods:
- forward_backward(epsilon, theta, local, pi, sprobs): the computation
"""

class FastRestrictedHMM:
    def __init__(self):
        self.topics_ = self.states_ = 0


    def forward_backward(self, epsilon, theta, local, pi, sprobs):
        self.topics_ = len(theta)
        self.states_ = 2 * self.topics_

        assert(len(pi) == self.states_)
        assert(len(sprobs) == len(local))
        for i in range(len(local)):
            assert(len(sprobs[i]) == self.states_)
            assert(len(local[i]) == self.topics_)

        norm_factor = np.zeros(len(local))
        alpha = np.zeros((len(local), self.states_))
        beta = np.zeros((len(local), self.states_))

        self.init_alpha(pi, local[0], norm_factor, alpha[0])
        self.compute_all_alphas(local, theta, epsilon, norm_factor, alpha)
        self.init_beta(norm_factor[-1], beta[-1])
        self.compute_all_betas(local, theta, epsilon, norm_factor, beta)
        self.combine_all_probs(alpha, beta, sprobs)
        return np.log(norm_factor).sum() # this is compute_loglik


    def init_alpha(self, pi, local0, norm_factor, alpha0):
        for i in range(self.topics_):
            alpha0[i] = local0[i] * pi[i]
            alpha0[i+self.topics_] = local0[i] * pi[i+self.topics_]
        norm_factor[0] = alpha0.sum()
        alpha0 /= norm_factor[0]


    def compute_all_alphas(self, local, theta, epsilon, norm_factor, alpha):
        for i in range(1, len(local)-1):
            for s in range(self.topics_):
                alpha[i, s] = epsilon * theta[i] * local[i, s]
                alpha[i, s+self.topics_] = (1-epsilon) * (alpha[i-1, s] + alpha[i-1, s+self.topics_]) * local[i, s]
            norm_factor[i] = alpha[i].sum()
            alpha[i] /= norm_factor[i]


    def init_beta(self, norm, beta_t_1):
        for i in range(self.states_):
            beta_t_1[i] = 1
        beta_t_1 /= norm


    def compute_all_betas(self, local, theta, epsilon, norm_factor, beta):
        for i in range(len(local)-2, -1, -1):
            trans_sum = 0.0
            for s in range(self.topics_):
                trans_sum += epsilon * theta[s] * local[i+1, s] * beta[i+1, s]

            for s in range(self.topics_):
                beta[i, s] = trans_sum + (1-epsilon) * local[i+1, s] * beta[i+1, s]
                beta[i, s+self.topics_] = beta[i, s]
            beta[i] /= norm_factor[i]


    def combine_all_probs(self, alpha, beta, sprobs):
        for i in range(len(alpha)):
            for s in range(self.states_):
                sprobs[i, s] = alpha[i, s] * beta[i, s]
            sprobs[i] /= sprobs[i].sum()
