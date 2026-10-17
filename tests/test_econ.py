"""Indicator dictionary and industry graph: schema, referential integrity, traversal, panel. Offline."""
import pandas as pd
import pytest

from mlab.econ import graph as eg
from mlab.econ.graph import build, naics_sector, section_of

ENUM = {"access": {"free", "free_key", "freemium", "paid", "members"},
        "commercial_use": {"yes", "no", "conditional", "unverified"},
        "storage_allowed": {"yes", "no", "conditional", "unverified"}}


@pytest.fixture(scope="module")
def g():
    return build()


def test_loads_without_problems(g):
    assert g.problems == []
    s = g.summary()["nodes"]
    assert s["ind"] >= 50 and s["sig"] >= 140 and s["kpi"] >= 60 and s["src"] >= 75 and s["mkt"] >= 50


def test_every_edge_endpoint_exists(g):
    for e in g.edges:
        assert e.src in g.nodes and e.dst in g.nodes, (e.src, e.type, e.dst)


def test_source_fields_use_the_enums(g):
    for n in g.of_kind("src"):
        for k, allowed in ENUM.items():
            assert n.data[k] in allowed, (n.id, k, n.data[k])
        assert isinstance(n.data["verified"], bool)


def test_every_series_points_at_a_registered_source(g):
    for n in g.of_kind("sig"):
        for s in n.data["series"]:
            assert f"src:{s['source']}" in g.nodes, (n.id, s)
            if s.get("via"):
                assert f"src:{s['via']}" in g.nodes, (n.id, s)


def test_explicit_edges_carry_evidence(g):
    for e in g.edges:
        if e.type in ("drives", "leads"):
            assert e.attrs["evidence"] in ("literature", "prior", "tested")
            assert e.attrs.get("sign") in ("+", "-")
            lo, hi = map(int, e.attrs["lag"].split("-"))
            assert 0 <= lo <= hi


def test_industry_tree_is_a_tree(g):
    for n in g.of_kind("ind"):
        seen, cur = set(), n.key
        while cur:
            assert cur not in seen, f"cycle at {cur}"
            seen.add(cur)
            cur = g.nodes[f"ind:{cur}"].data.get("parent")
        assert "economy" in seen


def test_section_and_sector_helpers():
    assert section_of("47.52") == "G" and section_of("49") == "H" and section_of("C") == "C"
    assert section_of("01.30") == "A" and section_of("99") == "U"
    assert naics_sector("444110") == "44-45" and naics_sector("484122") == "48-49" and naics_sector("23") == "23"


def test_classification_codes_roll_up(g):
    assert any(e.type == "child_of" and e.dst == "cls:naics:44-45" for e in g.out["cls:naics:444110"])
    assert any(e.type == "maps_to" and e.dst == "cls:isic:G" for e in g.out["cls:uksic:G"])


def test_resolve(g):
    assert g.resolve("garden_ornaments") == "ind:garden_ornaments"
    assert g.resolve("sig:us_saving_rate") == "sig:us_saving_rate"
    with pytest.raises(KeyError):
        g.resolve("no-such-node-anywhere")


def test_walk_and_path(g):
    near = {nid for _, _, nid, _ in g.walk("ind:pallet_networks", depth=1)}
    assert {"ind:road_freight", "ind:builders_merchants", "sig:uk_pallet_network_volumes"} <= near
    p = g.path("sig:uk_mortgage_approvals", "ind:retail_tiles_flooring")
    assert p and "sig:uk_mortgage_approvals" in (p[0].src, p[0].dst)
    assert g.path("ind:economy", "ind:economy") == []


def test_walk_filters_types_and_direction(g):
    out = g.walk("ind:sanitaryware_mfg", depth=1, types={"input_to"}, direction="out")
    assert {nid for *_, nid, _ in out} == {"ind:builders_merchants", "ind:retail_bathroom_online"}


def test_sideways_finds_same_kind_peers(g):
    peers = g.sideways("src:fred", hops=2)
    assert peers and all(g.nodes[p].kind == "src" for p, _, _ in peers)
    assert "src:fred" not in {p for p, _, _ in peers}
    assert peers[0][1] == 1.0  # normalised
    assert "src:census" in {p for p, _, _ in peers[:3]}
    imports = {p for p, _, _ in g.sideways("sig:uk_imports_garden_ornaments")}
    assert "sig:us_imports_garden_ornaments" in imports


def test_sources_for_covers_mirrors(g):
    src = g.sources_for("sig:us_saving_rate")
    assert "src:fred" in src and "src:bea" in src


def test_frames_for_network_export(g):
    nodes, edges = g.frames()
    assert {"id", "kind", "name"} <= set(nodes.columns)
    assert {"src", "dst", "type", "sign", "lag", "evidence"} <= set(edges.columns)
    assert len(edges) == len(g.edges)


def test_docs_render(g):
    from mlab.econ.docs import render
    md = render(g)
    for h in ("## Industry tree", "## Indicators", "## Sources", "## Unit-economics KPIs", "## Economic links"):
        assert h in md


def _fake_series(spec):
    idx = pd.date_range("2000-01-01", "2024-12-01", freq="MS")
    s = pd.Series(range(1, len(idx) + 1), index=idx, dtype=float)
    s.attrs["source"] = f"fake {spec['id']}"
    return s


def test_panel_builds_and_respects_storage(g, monkeypatch, tmp_path):
    from mlab.econ import fetch
    monkeypatch.setattr(fetch, "fetch_series", _fake_series)
    df, prov = fetch.panel(g, "ind:retail_home_improvement", depth=1, geo="UK", target=None, log=lambda *a: None)
    assert len(df.columns) >= 3 and set(prov["source"]) == {"ons"}
    out = tmp_path / "uk.csv"
    fetch.panel(g, "ind:retail_home_improvement", depth=1, geo="UK", target=None, out=str(out), log=lambda *a: None)
    assert out.exists()
    with pytest.raises(PermissionError):  # FRED's terms forbid storing its content
        fetch.panel(g, "sig:us_saving_rate", depth=0, target=None, out=str(tmp_path / "us.csv"), log=lambda *a: None)


def test_transforms():
    from mlab.econ.fetch import _transform
    s = pd.Series([100.0] * 12 + [110.0], index=pd.date_range("2020-01-31", periods=13, freq="ME"))
    assert _transform(s, "yoy", "M").iloc[-1] == pytest.approx(10.0)
    assert _transform(s, "diff", "M").iloc[-1] == pytest.approx(10.0)


def test_build_reports_bad_references(tmp_path):
    import shutil
    for f in eg.DATA.glob("*.yaml"):
        shutil.copy(f, tmp_path / f.name)
    with open(tmp_path / "edges.yaml", "a") as f:
        f.write("  - {from: sig:us_saving_rate, to: ind:nowhere, type: drives, sign: '+', lag: '0-1', evidence: prior}\n")
    assert any("ind:nowhere" in p for p in build(tmp_path).problems)
