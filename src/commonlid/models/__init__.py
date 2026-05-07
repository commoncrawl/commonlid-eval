"""LID model implementations. Importing this package registers every model.

Submodule imports fire the ``@register_model`` decorators. Order is
deterministic so ``list_models()`` output stays stable.

:class:`~commonlid.models.dspy_llm.DSPyLLMModel` is intentionally not
auto-registered because it needs per-instance configuration; import it
directly if you want to evaluate an LLM.
"""

from commonlid.models import afrolid as _afrolid  # noqa: F401
from commonlid.models import cld2 as _cld2  # noqa: F401
from commonlid.models import cld3 as _cld3  # noqa: F401
from commonlid.models import fasttext_ft as _fasttext_ft  # noqa: F401
from commonlid.models import funlangid as _funlangid  # noqa: F401
from commonlid.models import glotlid as _glotlid  # noqa: F401
from commonlid.models import openlidv2 as _openlidv2  # noqa: F401
from commonlid.models import pyfranc as _pyfranc  # noqa: F401
from commonlid.models.dspy_llm import DSPyLLMModel

__all__ = ["DSPyLLMModel"]
