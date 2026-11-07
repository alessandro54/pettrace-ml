from tests import ROOT


def test_unit_tree_mirrors_source_modules():
    unit_root = ROOT / "tests/unit/pettrace"
    source_root = ROOT / "src/pettrace"
    for test in unit_root.rglob("test_*.py"):
        relative = test.relative_to(unit_root)
        source = source_root / relative.parent / relative.name.removeprefix("test_")
        assert source.is_file(), f"{test.relative_to(ROOT)} does not mirror {source.relative_to(ROOT)}"


def test_no_tests_live_in_the_legacy_flat_root():
    assert not list((ROOT / "tests").glob("test_*.py"))
