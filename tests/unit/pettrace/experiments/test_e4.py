from pettrace.experiments.e4 import _query_signature


def test_query_signature_ignores_runtime_cache_root():
    row = {"photo": "/old/cache/cat/photo.jpg", "animal": "cat-1",
           "species": "cat", "gallery_animals": 12}
    moved = {**row, "photo": "/new/runtime/cat/photo.jpg"}
    assert _query_signature([row]) == _query_signature([moved])
