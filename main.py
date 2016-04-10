from collections import defaultdict
from gensim.models import Word2Vec
import re
from sklearn.cluster import AffinityPropagation
from nltk import tokenize
from nltk.cluster import KMeansClusterer
import theano
import theano.tensor as T
import numpy as np


def load_ground_truth_clusters(path):
    with open(path, 'r') as f:
        clusters_content = f.read().splitlines()
    cluster_id = 0
    clusters = defaultdict(list)
    for line in clusters_content:
        if line != "##########" and line.strip():
            clusters[cluster_id].append(line)
        else:
            cluster_id += 1
    return clusters


def load_data(path):
    with open(path, 'r') as f:
        return f.read().splitlines()


def compute_stats(targets, predictions):
    pass


class SentenceToVec(object):
    def __init__(self, path_to_word2vec_model):
        self.model = Word2Vec.load(path_to_word2vec_model)

    def __call__(self, sentence):
        letters_only = re.sub('[^a-zA-Z]', ' ', sentence)
        words = letters_only.lower().split()
        vec = np.zeros(1000, dtype=theano.config.floatX)
        count = 0
        for word in words:
            try:
                vec += self.model[word]
                count += 1
            except KeyError:
                pass
        if count == 0:
            words = ''.join(words)
            for c in words:
                try:
                    vec += self.model[c]
                    count += 1
                except KeyError:
                    pass
        vec /= count
        return vec


class AngularDistance(object):
    def __init__(self):
        u = T.vector('u')
        v = T.vector('v')

        m = T.dot(u, v)

        norm_u = u.norm(2)
        norm_v = v.norm(2)

        denominator = T.dot(norm_u, norm_v)

        distance = 1 - m / denominator

        self.tf_distance = theano.function([u, v], distance)

    def __call__(self, u, v):
        return self.tf_distance(u, v)


if __name__ == "__main__":
    clusters = load_ground_truth_clusters('data/clusters.txt')
    # n_clusters = len(clusters.keys())
    n_clusters = 2
    data = load_data('data/lines.txt')

    sen2vec = SentenceToVec('data/en.model')

    angular_dist = AngularDistance()

    representation_per_line = [sen2vec(line) for line in data][:10]

    kmeans = KMeansClusterer(n_clusters, angular_dist, repeats=5)

    clusters = kmeans.cluster(representation_per_line, True)

    print clusters
