# naive-bayes-spam-filter

Multinomial Naive Bayes written from scratch (Laplace smoothing, log probabilities), trained on 18 hand-written messages and tested on 6 held-out ones.

## Run

```bash
pip install -r requirements.txt
python demo.py
pytest
```

## Output (reproduced)

```
$ python3 demo.py
spam  -> spam  free entry win cash prize
spam  -> spam  act now limited offer cash
spam  -> ham   you have been selected for a free reward
ham   -> ham   the training run completed overnight
ham   -> ham   dinner at my place on friday
ham   -> ham   attached is the draft of the paper
accuracy: 5/6

$ pytest -q
4 passed in 0.01s
```

## Limits

- Tiny hand-written dataset. One held-out spam message is misclassified, as the output shows. Not a real spam filter.
