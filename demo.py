from nb import NaiveBayes
from data import TRAIN, TEST

m = NaiveBayes().fit([t for t, _ in TRAIN], [y for _, y in TRAIN])
ok = 0
for t, y in TEST:
    p = m.predict(t)
    ok += p == y
    print(f"{y:<5} -> {p:<5} {t}")
print(f"accuracy: {ok}/{len(TEST)}")
