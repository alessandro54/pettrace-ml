import pytest

from pettrace.models.cuda import _plugin_spec


def test_cuda_plugin_support_is_explicit_and_pinned():
    assert _plugin_spec("12.8")[0] == "12"
    assert _plugin_spec("13.0")[0] == "13"
    with pytest.raises(RuntimeError, match="CUDA 12.8"):
        _plugin_spec("12.7")
