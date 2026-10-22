"""Economic indicator dictionary and industry graph: macro aggregates down to product lines and company
KPIs, every source with its access and licence, wired as a typed graph for traversal and ML features.

    from mlab.econ import load
    g = load()
    g.walk("ind:garden_ornaments", depth=2)
    g.sideways("src:ons")
"""
from .graph import EconGraph, load  # noqa: F401
