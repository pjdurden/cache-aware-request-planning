import pathlib
import pytest
from ccrp.eval import Point

mpl = pytest.importorskip("matplotlib")
from ccrp.figures import pareto_plane


def test_pareto_plane_writes_png(tmp_path):
    points = [
        Point("naive", 1.0, 0.0, 0.0),
        Point("cache_shaping", 0.6, 0.0, 0.0),
        Point("compression", 0.5, 0.1, 0.15),
        Point("semantic_cache", 0.7, 0.2, 0.1),
    ]
    out = tmp_path / "pareto.png"
    result = pareto_plane(points, str(out))
    assert result == str(out)
    assert pathlib.Path(result).exists()
    assert out.stat().st_size > 0
