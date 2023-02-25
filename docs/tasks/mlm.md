Masked Language Modeling
========================

The task proposed *inter alios* by {cite:ts}`devlin2018BERTPretrainingDeep` for pretraining of
bidirectional encoders. It consists in training a model to denoise a sequence, where the noise
preserves the sentence length, most famously by *masking* some words:

- **Original sentence**: “The little tabby cat is happy”
- **Masked sentence**: “The little \<MASK\> cat is happy”

The masked sentence serves as input and the expected output of the model is the original. For
transformers model, it usually entails using an encoder-only architecture and using a thin word
predictor “head” (for instance a linear layer) on top of its last hidden states.
{cite:ts}`muller2022BERT101State` gives a good general introduction to these models.

More specifically, following e.g. the pretraining procedure of RoBERTa pretraining
{cite}`liu2019RoBERTaRobustlyOptimized`, in Zelda Rose, we apply two types of to the input
sentences :

- **Masking**: replacing some tokens with a mask token, as in the example above.
- **Switching** replacing some tokens with a mask token by replacing them with another random token
  from the vocabulary.

## Task parameters

```python
change_ratio: float = 0.15
mask_ratio: float = 0.8
switch_ratio: float = 0.1
```

- `change_ratio` is the proportion of tokens to which we apply some change either masking or
  switching.
- `mask_ratio` is the proportion of tokens targetted for a change that are replaced by a mask
  token.
- `switch_ratio` is the proportion of tokens targetted for a change that are replaced by a
  random non-mask token

Note that all of these should be floats between `0.0` and `1.0` and that you should have
`mask_ratio+switch_ratio <= 1.0` too.

## Bibliography

```{bibliography}
:filter: docname in docnames
:style: alpha
```
