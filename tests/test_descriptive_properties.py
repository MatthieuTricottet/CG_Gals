import numpy as np
import pandas as pd

from src import descriptive_properties as dp


def _synthetic_frame(seed: int = 3) -> pd.DataFrame:
    """Four samples of quartets; CG4 selections 1 and 2 share one host."""

    rng = np.random.default_rng(seed)
    rows = []
    for sample in dp.SAMPLES:
        for group in range(1, 9):
            physical = (
                "Lim:100" if sample == "CG4" and group in (1, 2) else f"Lim:{group}"
            )
            for rank in range(1, 5):
                mass = (11.1 if rank == 1 else 10.3) + rng.normal(0, 0.15)
                rows.append(
                    {
                        "sample": sample,
                        "Group": group,
                        "group_uid": f"{sample}:{group}",
                        "physical_group": physical,
                        "rank": rank,
                        "logMstar": mass,
                        "h_alpha_eqw": rng.normal(-3, 3),
                        "log_NII_Ha": rng.normal(-0.2, 0.3),
                        "log_OIII_Hb": rng.normal(0.0, 0.4),
                        "Dn4000": rng.normal(1.7, 0.2),
                        "log_Rchl_r_kpc": rng.normal(0.5, 0.1),
                        "log_petroR50_kpc": rng.normal(0.55, 0.1),
                    }
                )
    return pd.DataFrame(rows)


def test_scope_filter_selects_ranks():
    frame = _synthetic_frame()
    trends = dp.compute_property_trends(frame)
    assert set(trends["scope"]) == {"all", "bggs", "satellites"}
    counts = trends.loc[trends["panel"] == "dn4000"].groupby(["scope", "sample"])[
        "n_galaxies"
    ].sum()
    for sample in dp.SAMPLES:
        part = frame.loc[frame["sample"] == sample]
        assert counts[("bggs", sample)] == int(part["rank"].eq(1).sum())
        assert counts[("satellites", sample)] == int(part["rank"].gt(1).sum())
        assert counts[("all", sample)] == len(part)


def test_bootstrap_blocks_on_physical_group():
    frame = _synthetic_frame()
    work = dp._prepare(frame)
    assert (work["Group"] == work["physical_group"]).all()
    trends = dp.compute_property_trends(frame)
    cg_bgg = trends.loc[
        (trends["panel"] == "dn4000")
        & (trends["scope"] == "bggs")
        & (trends["sample"] == "CG4")
    ]
    # 8 CG4 selections, two of which share a host: 7 physical blocks
    assert int(cg_bgg["n_groups"].sum()) == 7
    assert int(cg_bgg["n_galaxies"].sum()) == 8


def test_rank_split_figure_renders_eight_panels(tmp_path, monkeypatch):
    frame = _synthetic_frame()
    trends = dp.compute_property_trends(frame)
    seen = {}
    original = dp.finish_binned_figure

    def capture(fig, axes, filename, ylabel="Fraction"):
        seen["shape"] = axes.shape
        seen["titles"] = [ax.get_title() for ax in axes.flat]
        seen["legend"] = axes[0, 0].get_legend_handles_labels()[1]
        original(fig, axes, filename, ylabel=ylabel)

    monkeypatch.setattr(dp, "finish_binned_figure", capture)
    path = tmp_path / "fig_property_trends.pdf"
    dp.plot_property_trends(trends, str(path))
    assert path.is_file() and path.stat().st_size > 0
    assert seen["shape"] == (4, 2)
    assert seen["titles"][0].startswith("(a) BGGs:")
    assert seen["titles"][1].startswith("(b) Satellites:")
    assert len(seen["legend"]) == len(dp.SAMPLES)
