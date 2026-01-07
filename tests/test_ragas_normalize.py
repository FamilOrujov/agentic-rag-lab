from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


def _load_ragas_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "ragas_eval_langfuse.py"
    spec = spec_from_file_location("ragas_eval_langfuse", path)
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def test_normalize_to_str_list() -> None:
    mod = _load_ragas_module()
    normalize = mod.normalize_to_str_list

    assert normalize(None) == []
    assert normalize("hi") == ["hi"]
    assert normalize({"text": "x"}) == ["x"]
    assert normalize({"page_content": "y"}) == ["y"]
    assert normalize(["a", 2]) == ["a", "2"]
    assert normalize((1, "b")) == ["1", "b"]
    assert normalize(b"hi") == ["hi"]
