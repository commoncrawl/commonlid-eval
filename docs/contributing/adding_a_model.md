# Adding a model to the leaderboard

The CommonLID leaderboard is available [here](https://huggingface.co/spaces/commoncrawl/commonlid).

1. Add the [model implementation](#adding-a-model-implementation) to `commonlid`
2. [Evaluate](#evaluate-new-model) the desired model using `commonlid` on the benchmarks
3. Push the results to the [results repository](https://huggingface.co/datasets/commoncrawl/commonlid-results) via a PR. Once merged they will appear on the leaderboard.

## Requesting an evaluation

If you want a model to be evaluated but are not submitting the results yourself, open an issue instead and provide the required information.

## Adding a model implementation

Adding a model implementation to `commonlid` is quite straightforward. Typically, it only requires that you provide the text-to-language prediction method and add it to the [model directory](https://github.com/commoncrawl/commonlid-eval/tree/main/src/commonlid/models):

```python
# src/commonlid/models/my_model.py
from collections.abc import Sequence

from commonlid.core.lid_model import LIDModel
from commonlid.core.registry import get_model, register_model


@register_model
class MyModel(LIDModel):
    model_id = "my_model"

    def _predict_batch(self, texts: Sequence[str]) -> list[str | None]:
        # Return one ISO 639-3 code (or None for undetermined) per input.
        # `texts` arrives post-OpenLID-normer cleaning by default;
        # set `requires_preprocessing = False` to receive raw text.
        return ["eng"] * len(texts)


assert get_model("my_model").predict(["hi"]) == ["eng"]
```

Then import it from `src/commonlid/models/__init__.py` so the decorator
fires on `import commonlid`:

```python
from commonlid.models import my_model as _my_model  # noqa: F401
```


### Reporting a confidence

`_predict_batch` may return a `LIDPrediction(iso639_3, score)` in place of a
bare code whenever the backend reports a confidence. The two forms mix freely
within one batch, and `predict()` keeps returning bare codes either way.

```python
from commonlid.core.lid_model import LIDModel, LIDPrediction

    def _predict_batch(self, texts: Sequence[str]) -> list[LIDPrediction]:
        return [LIDPrediction("eng", 0.97) for _ in texts]
```

The score reaches `predictions.jsonl` and `commonlid predict` output. Only
pass a real confidence: a constant, or a number on an unbounded scale, is
worse than `None` because it reads like one. See the table in the README for
what each shipped model does.

### Reporting what a call costs

Models billed per call declare their price list and implement two optional
hooks, so that `commonlid estimate-cost` can budget a run before it starts:

```python
from commonlid.cost import Range, RateCard

    pricing = RateCard(
        card_id="my-api",
        rates={"characters": 20.0 / 1_000_000},  # USD per unit
        as_of="2026-10-02",
        source="https://example.com/pricing",
    )

    def _estimate_usage(self, texts: Sequence[str]) -> dict[str, float]:
        # Billable usage of these (already preprocessed) texts, counted offline.
        return {"characters": float(sum(len(t) for t in texts))}

    def usage_assumptions(self) -> dict[str, Range]:
        # Per-sample usage that can't be counted offline, as low/expected/high.
        return {"output_tokens": Range(12, 16, 32)}
```

When the price depends on the instance, as for LLMs priced by model name,
override `rate_card()` instead of setting `pricing`.

To support `--calibrate`, override `measure_usage(texts)` as well. It should
predict `texts` for real and return the usage the API reported for each
request. Always give `pricing` an `as_of` date and a source URL, since prices
drift. Local models need none of this: their cost
is compute time, which `--hardware` and `--throughput` cover.

### Adding model dependencies

If you are adding a model that requires additional dependencies, you can add them to the `pyproject.toml` file, under optional dependencies:

```toml
cld3 = ["cld3-py>=3.1"]
```

This ensures that the implementation does not break if a package is updated.

As it is an optional dependency, you can't use top-level dependencies, but will instead have to use import inside the wrapper scope.

## Evaluate new model

As soon as the model implementation is registered, you can run this command to evaluate your model on CommonLID and its nano version:

```bash
commonlid run \
  --model my_model \
  --dataset commonlid --dataset commonlid_nano \
  --output-dir ./data/results
```

You may indeed reinstall the `commonlid` package with your changes if the package was not installed in editable mode.

## Uploading the results data (PR-based)

After running the evaluation locally, you can upload the results to our [HF results repository](https://huggingface.co/datasets/commoncrawl/commonlid-results) as follows:

```bash
hf auth login                                   # token with write access to the results dataset
make leaderboard-upload                         # opens a Pull Request from ./data/results
# Override the target with: make leaderboard-upload LEADERBOARD_REPO=other/repo LEADERBOARD_DIR=./elsewhere
# Optional: pass --skip-predictions via `uv run commonlid leaderboard upload ...` directly.
```

The CLI always opens a Pull Request rather than pushing to the default
branch, so the dataset owner reviews before merging.
