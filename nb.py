"""Multinomial Naive Bayes text classifier from scratch (Laplace smoothing, log probabilities)."""
import math
import re
from collections import Counter, defaultdict


def tokenize(t):
    return re.findall(r"[a-z0-9']+", t.lower())


class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, texts, labels):
        self.counts = defaultdict(Counter)
        self.docs = Counter(labels)
        for t, y in zip(texts, labels):
            self.counts[y].update(tokenize(t))
        self.vocab = {w for c in self.counts.values() for w in c}
        self.total = {y: sum(c.values()) for y, c in self.counts.items()}
        return self

    def log_scores(self, text):
        n = sum(self.docs.values())
        out = {}
        for y in self.docs:
            s = math.log(self.docs[y] / n)
            for w in tokenize(text):
                if w in self.vocab:
                    s += math.log((self.counts[y][w] + self.alpha) / (self.total[y] + self.alpha * len(self.vocab)))
            out[y] = s
        return out

    def predict(self, text):
        s = self.log_scores(text)
        return max(s, key=s.get)
