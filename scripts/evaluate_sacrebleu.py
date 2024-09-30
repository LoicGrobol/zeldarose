import json
import pathlib
from typing import cast, Iterable, TextIO

import click
import jsonlines
import transformers

from rich.progress import track
from sacrebleu.metrics import bleu, chrf, ter


@click.command(
    help=(
        "Apply a mBART translation model to a series of testcases for quantitative and eyeballing"
        " purposes"
    )
)
@click.argument("model")
@click.argument("testcases", type=click.File("r"))
@click.argument(
    "output_dir",
    type=click.Path(file_okay=False, dir_okay=True, writable=True, path_type=pathlib.Path),
)
@click.option("--model-src-langcode", show_default=True)
@click.option("--model-tgt-langcode", show_default=True)
@click.option("--src-langcode", default="br", show_default=True)
@click.option("--tgt-langcode", default="fr", show_default=True)
def main(
    model: str,
    testcases: TextIO,
    model_src_langcode: str | None,
    model_tgt_langcode: str | None,
    output_dir: pathlib.Path,
    src_langcode: str,
    tgt_langcode: str,
):
    output_dir.mkdir(exist_ok=True, parents=True)
    if model_src_langcode is None:
        model_src_langcode = src_langcode
    if model_tgt_langcode is None:
        model_tgt_langcode = tgt_langcode
    pipeline = transformers.pipeline("translation", model=model)
    with jsonlines.Reader(testcases) as in_stream:
        samples = [
            line.get("translation", line)
            for line in track(in_stream, description="Reading corpus…")
        ]
    predictions = pipeline(
        track([s[src_langcode] for s in samples], description="Translating…"),
        src_lang=model_src_langcode,
        tgt_lang=model_tgt_langcode,
    )
    result = [
        {
            "source": s[src_langcode],
            "target": s[tgt_langcode],
            "prediction": p[0]["translation_text"],
        }
        for s, p in zip(samples, cast(Iterable, predictions))
    ]

    with jsonlines.open(output_dir / "predictions.jsonl", "w") as out_stream:
        out_stream.write_all(result)

    predictions = [row["prediction"] for row in result]
    references = [row["target"] for row in result]

    metrics = dict()
    for m_name, m in (("bleu", bleu.BLEU()), ("chrf", chrf.CHRF()), ("ter", ter.TER())):
        metrics[m_name] = {
            "value": m.corpus_score(predictions, [references]).score,
            "signature": str(m.get_signature()),
        }
    with (output_dir / "metrics.json").open("w") as out_stream:
        json.dump(metrics, out_stream, indent=4)


if __name__ == "__main__":
    main()
