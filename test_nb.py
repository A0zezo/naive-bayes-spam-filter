from nb import NaiveBayes, tokenize
from data import TRAIN, TEST


def model():
    return NaiveBayes().fit([t for t, _ in TRAIN], [y for _, y in TRAIN])


def test_tokenize():
    assert tokenize("Free CASH!!") == ["free", "cash"]


def test_obvious_spam_and_ham():
    m = model()
    assert m.predict("free cash prize") == "spam"
    assert m.predict("review the project report") == "ham"


def test_prior_used_for_unknown_words():
    assert model().predict("qqq www") in {"spam", "ham"}


def test_heldout_accuracy():
    m = model()
    assert sum(m.predict(t) == y for t, y in TEST) / len(TEST) >= 0.75
