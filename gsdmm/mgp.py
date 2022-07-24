import logging

import numpy as np
from numpy.random import multinomial
from numpy import argmax, log, exp

logging.basicConfig(format='%(asctime)s - %(filename)s:%(lineno)d - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieGroupProcess:
    def __init__(self, K=8, alpha=0.1, beta=0.1, n_iters=30):
        '''
        A MovieGroupProcess is a conceptual model introduced by Yin and Wang 2014 to
        describe their Gibbs sampling algorithm for a Dirichlet Mixture Model for the
        clustering short text documents.
        Reference: http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf

        Imagine a professor is leading a film class. At the start of the class, the students
        are randomly assigned to K tables. Before class begins, the students make lists of
        their favorite films. The teacher reads the role n_iters times. When
        a student is called, the student must select a new table satisfying either:
            1) The new table has more students than the current table.
        OR
            2) The new table has students with similar lists of favorite movies.

        :param K: int
            Upper bound on the number of possible clusters. Typically many fewer
        :param alpha: float between 0 and 1
            Alpha controls the probability that a student will join a table that is currently empty
            When alpha is 0, no one will join an empty table.
        :param beta: float between 0 and 1
            Beta controls the student's affinity for other students with similar interests. A low beta means
            that students desire to sit with students of similar interests. A high beta means they are less
            concerned with affinity and are more influenced by the popularity of a table
        :param n_iters:
        '''
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.n_iters = n_iters

        # slots for computed variables
        self.number_docs = None
        self.vocab_size = None
        self.word_to_idx = None
        self.idx_to_word = None
        self.cluster_doc_count = np.zeros(shape=(K,), dtype=np.uintc)
        self.cluster_word_count = np.zeros(shape=(K,), dtype=np.uintc)
        self.cluster_word_distribution = None
        self.document_words = None
        self.document_word_indices = {}
        self.document_size_vectors = {}
        self.sampling_distribution = [1/K for _ in range(K)]

    @staticmethod
    def from_data(K, alpha, beta, D, vocab_size, cluster_doc_count, cluster_word_count, cluster_word_distribution):
        '''
        Reconstitute a MovieGroupProcess from previously fit data
        :param K:
        :param alpha:
        :param beta:
        :param D:
        :param vocab_size:
        :param cluster_doc_count:
        :param cluster_word_count:
        :param cluster_word_distribution:
        :return:
        '''
        mgp = MovieGroupProcess(K, alpha, beta, n_iters=30)
        mgp.number_docs = D
        mgp.vocab_size = vocab_size
        mgp.cluster_doc_count = cluster_doc_count
        mgp.cluster_word_count = cluster_word_count
        mgp.cluster_word_distribution = cluster_word_distribution
        return mgp

    @staticmethod
    def _sample(p):
        '''
        Sample with probability vector p from a multinomial distribution
        :param p: list
            List of probabilities representing probability vector for the multinomial distribution
        :return: int
            index of randomly selected output
        '''
        return argmax(multinomial(1, p))

    @staticmethod
    def _create_index_maps(docs: list[list[str]]) -> tuple[dict[str, int], dict[int, str]]:
        unique_strings = sorted(set().union(*map(set, docs)))
        idx_to_str = {}
        str_to_idx = {}
        for idx, key in enumerate(unique_strings):
            idx_to_str[idx] = key
            str_to_idx[key] = idx
        return str_to_idx, idx_to_str

    def fit(self, docs, vocab_size):
        '''
        Cluster the input documents
        :param docs: list of list
            list of lists containing the unique token set of each document
        :param V: total vocabulary size for each document
        :return: list of length len(doc)
            cluster label for each document
        '''
        alpha, beta, K, n_iters, V = self.alpha, self.beta, self.K, self.n_iters, vocab_size
        self.word_to_idx, self.idx_to_word = self._create_index_maps(docs)

        D = len(docs)
        self.number_docs = D
        self.vocab_size = vocab_size

        # Init word arrays
        #  doc * word matrix
        docs_words = np.zeros(shape=(D, len(self.idx_to_word)), dtype=np.uintc)
        for doc_idx, doc in enumerate(docs):
            doc_word_indices = [self.word_to_idx[word] for word in doc]
            np.add.at(docs_words[doc_idx], doc_word_indices, 1)
            self.document_word_indices[doc_idx] = doc_word_indices
            self.document_size_vectors[doc_idx] = np.array(range(len(doc)), dtype=np.uintc).reshape(1, len(doc))
        self.document_words = docs_words

        #  cluster * word matrix
        n_z_w = np.zeros(shape=(K, len(self.idx_to_word)))
        self.cluster_word_distribution = n_z_w

        # unpack to easy var names
        m_z, n_z = self.cluster_doc_count, self.cluster_word_count
        cluster_count = K
        d_z = [None for i in range(len(docs))]

        # initialize the clusters
        for i, doc in enumerate(docs):

            # choose a random  initial cluster for the doc
            z = self._sample(self.sampling_distribution)
            d_z[i] = z
            m_z[z] += 1
            n_z[z] += len(doc)
            n_z_w[z] += docs_words[i]

        for _iter in range(n_iters):
            total_transfers = 0

            for i, doc in enumerate(docs):

                # remove the doc from it's current cluster
                z_old = d_z[i]

                m_z[z_old] -= 1
                n_z[z_old] -= len(doc)
                n_z_w[z_old] -= docs_words[i]

                # draw sample from distribution to find new cluster
                p = self.score(doc, i)
                z_new = self._sample(p)

                # transfer doc to the new cluster
                if z_new != z_old:
                    total_transfers += 1

                d_z[i] = z_new
                m_z[z_new] += 1
                n_z[z_new] += len(doc)
                n_z_w[z_new] += docs_words[i]

            cluster_count_new = np.count_nonzero(m_z)
            logging.info(
                f"iter {_iter}: transferred {total_transfers} documents with {cluster_count_new} clusters populated"
            )
            if total_transfers == 0 and cluster_count_new == cluster_count and _iter>25:
                logging.info("Converged.  Breaking out.")
                break
            cluster_count = cluster_count_new
        self.cluster_word_distribution = n_z_w
        return d_z

    def score(self, doc: list[str], doc_idx: int) -> np.ndarray:
        '''
        Score a document

        Implements formula (3) of Yin and Wang 2014.
        http://dbgroup.cs.tsinghua.edu.cn/wangjy/papers/KDD14-GSDMM.pdf

        :param doc: list[str]: The doc token stream
        :return: list[float]: A length K probability vector where each component represents
                              the probability of the document appearing in a particular cluster
        '''
        alpha, beta, K, V, D = self.alpha, self.beta, self.K, self.vocab_size, self.number_docs
        m_z, n_z, n_z_w = self.cluster_doc_count, self.cluster_word_count, self.cluster_word_distribution
        word_indices = self.document_word_indices[doc_idx]
        doc_size_vector = self.document_size_vectors[doc_idx]
        #  We break the formula into the following pieces
        #  p = N1*N2/(D1*D2) = exp(lN1 - lD1 + lN2 - lD2)
        #  lN1 = log(m_z[z] + alpha)
        #  lN2 = log(D - 1 + K*alpha)
        #  lN2 = log(product(n_z_w[w] + beta)) = sum(log(n_z_w[w] + beta))
        #  lD2 = log(product(n_z[d] + V*beta + i -1)) = sum(log(n_z[d] + V*beta + i -1))

        doc_size = len(doc)
        v_beta = V*beta

        lD1 = log(D - 1 + K * alpha)
        lN1 = log(m_z + alpha)
        lN2 = log(n_z_w[:, word_indices] + beta).sum(axis=1)

        n_z_beta = n_z + beta
        lD2 = log(n_z_beta.reshape(K, 1) + doc_size_vector).sum(axis=1)

        p = exp(lN1 - lD1 + lN2 - lD2)

        # normalize the probability vector
        pnorm = sum(p)
        pnorm = pnorm if pnorm>0 else 1
        return p / pnorm

    def choose_best_label(self, doc):
        '''
        Choose the highest probability label for the input document
        :param doc: list[str]: The doc token stream
        :return:
        '''
        p = self.score(doc)
        return argmax(p),max(p)
