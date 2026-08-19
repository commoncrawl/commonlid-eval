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

For LLM (`dspy:`) models, estimate the token usage and cost before committing to
a full run:

```bash
commonlid estimate-tokens \
  --model dspy:openai/gpt-5 \
  --dataset commonlid \
  --reasoning-tokens 300
```

This makes no API calls — it tokenizes the reconstructed prompt with the model's
own tokenizer and prices it via LiteLLM. See the README's "Estimate tokens &
cost" section for details.

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
