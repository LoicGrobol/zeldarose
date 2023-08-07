mBART translation
=================

**NOTE** given the specific nature of the task, only models and tokenizers of the mBART/m2M100 family are allowed.

The task proposed by {cite:ts}`lewis2020BARTDenoisingSequencetoSequence`,
{cite:ts}`liu2020MultilingualDenoisingPretraining` and
{cite:ts}`tang2020MultilingualTranslationExtensible` for training text-to-text models. In our case, we
will mostly think of it as a translation task, but it could easily be adapted for other tasks such
as summarization. It consists of a pretraining an encoder-decoder for a self-supervised
**denoising** task, then fine-tuning it on a translation task, allowing to use non-parallel corpora
to improve machine translation.

{cite:ts}`lewis2020BARTDenoisingSequencetoSequence` experimented with several noise functions,
finally settling on *text infilling* and *sentence shuffling*. Since sentence shuffling assumes
*document-level* processing and Zelda Rose is meant for *sentence-level* training, we only implement
text infilling here, which consists of masking small spans of tokens with a single \<MASK\> token
each:

- **Original sentence**: “The little tabby cat is happy”
- **After infilling**: “The little \<MASK\> happy”

The masked sentence serves as input and the expected output of the model is the original. Since the
length of the target can not be easily be deduced from input, the models used for this task are
*encoder-decoders*, such as the original transformer model {cite}`vaswani2017AttentionAllYou`.

Translation is, as in {cite:ts}`vaswani2017AttentionAllYou`, also treated as a text-to-text task.

One innovation of Zelda Rose is that the models can also be trained *simultaneously* on denoising
and translation, with a weight hyperparameter that controls each task's contribution to the
optimized loss.

## Task parameters

```python
change_ratio: float = 0.3
denoise_langs: Optional[List[str]]
denoise_loss_ratio: float = 0.5
poisson_lambda: float = 3.0
source_langs: Optional[List[str]]
target_langs: Optional[List[str]]
strict_langs: bool = False
```

- `change_ratio` is the proportion of tokens to which we apply some change either masking or
  switching.
- `denoise_langs`, `source_langs` and `target_langs` are the codes for the languages in these
  respective roles. See below for their link with model and data format.
- `denoise_loss_ration` is the weight (between $0$ and $1$) given to the denoising loss in the
  multitask loss.
- `poisson_lambda` is the $λ$ parameter of the Poisson distribution from which the sizes of the
  masked spans are drawn
- `strict_langs` is a flag controlling if the lang codes are allowed to only partially match between
  dataset and model/tokenizer.


## Inputs and outputs

For this task, the train and dev datasets should be in the jsonlines format, every row being a mapping between langcode and translation in the corresponding language such as

```json
{"br": "Me am eus kanet", "fr": "J'ai chanté", "en": "I have sung"}
```

Or, for compatibility with 🤗 datasets, each row can be an arbitrary mapping, that has a `"translation"` key associated to a mapping in the previous format:

```json
{"translation": {"br": "Me am eus kanet", "fr": "J'ai chanté", "en": "I have sung"}}
```

Inputs can come either from local files or from a 🤗 dataset.

## Bibliography

```{bibliography}
:filter: docname in docnames
:style: alpha
```