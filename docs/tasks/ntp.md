Next token prediction
======================

The task also known as *Causal Language Modelling*, originating in
{cite:ts}`bengio2003NeuralProbabilisticLanguage`, used for first time using a transformer
architecture by {cite:ts}`howard2018UniversalLanguageModel` and popularized by
{cite:ts}`radford2019LanguageModelsAre` as a way to produce “universal learners”. It consists
simply, given the first $n$ tokens in a sentence, in predicting the $n+1$-th. Assuming a word-level
tokenization, input/output pairs thus look like this:

```python
{"input": ["Dr.", "Chef", "knew", "exactly", "where", "all", "of", "his"], "output": "feelings"}
```

In practice, when training, all the positions of a given sentence are predicted in the same batch,
thus improving the computational efficiency by reusing the hidden representations of the tokens.
This allows us to recast the task as a sentce-level token labelling task, where each token (except
the last one) in a sentence should be labelled with the token that follows it:

```python
{
  "sentence": ["Dr.", "Chef", "knew", "exactly", "where", "all", "of", "his", "feelings"],
  "labels": ["Chef", "knew", "exactly", "where", "all", "of", "his", "feelings", "were"]
}
```

Zelda Rose deals with these details itself, so the only thing you need to do as a user is to provide
a raw text dataset.

## Task parameters

No parameters for this task!

## Inputs and outputs

For this task, the train and dev datasets should be raw text, every line containing a single sample
(typically a sentence). It can come either from a local text file or from a 🤗 [text
dataset](https://huggingface.co/docs/datasets/nlp_load).

## Bibliography

```{bibliography}
:filter: docname in docnames
:style: alpha
```
