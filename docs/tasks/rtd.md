Replaced tokens detection
=========================

**NOTE** Since RTD requires training **two** models, the `--pretrained-model` and `--model-config`
arguments take two arguments (separated by a comma and no space, lie `--pretrained-model
lgrobol/deberta-minuscule_gen,lgrobol/deberta-minuscule_dis`). The first one goes to the generator
and the second one for the discriminator.

The task, proposed by {cite:ts}`clark2019ELECTRAPretrainingText` to train their ELECTRA model,
consists of training two antagonist neural networks in parallel. One, the *generator* is trained as
a masked language model, to fill in masked tokens in a sentence. The other, the *discriminator*, is
trained as a detector of replaced tokens. The hinge of the technique resides in the disparity
between the two models: the generator is in general much smaller than the discriminator. The
resulting ensemble can be made smaller (in terms of number of parameters) and trained faster than a
MLM model of equivalent performances.

Our implementation here follows more closely that of {cite:ts}`he2021DeBERTaV3ImprovingDeBERTa`, who extended the original ELECTRA to multilingualism and larger size, using tricks first presented by {cite:ts}`he2020DeBERTaDecodingenhancedBERT`.

A word of warning: the success in this task is heavily dependent on subtle hyperparameters choices. We have done our best to select reasonable default, but if you insist on using this pretraining, a comprehensive grid search might be in order (which- can defeat its frugality advantages).

## Task parameters

```python
discriminator_loss_weight: float = 1.0
embeddings_sharing: Optional[Literal["deberta", "electra"]] = None
mask_ratio: float = 0.15
```

- `discriminator_loss_weight` is the $α$ parameter for the combined loss
  $α\mathrm{loss}_{\mathrm{discriminator}}+\mathrm{loss}_{\mathrm{generator}}$ use to train the
  models.
- `embeddings_sharing` selects how the embedding layers of both models are shared: as in ELECTRA
  {cite}`clark2019ELECTRAPretrainingText`, as in DeBERTa v3 {cite}`he2021DeBERTaV3ImprovingDeBERTa`
  or not a all (`None`, the default).
- `mask_ratio` is the proportion of tokens that are masked for the generator.

## Inputs and outputs

For this task, the train and dev datasets should be raw text, every line containing a single sample
(typically a sentence). It can come either from a local text file or from a 🤗 [text
dataset](https://huggingface.co/docs/datasets/nlp_load).

## Bibliography

```{bibliography}
:filter: docname in docnames
:style: alpha
```
