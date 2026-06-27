import ccrp


def test_package_imports_and_has_version():
    assert isinstance(ccrp.__version__, str)
    assert ccrp.__version__ != ""
