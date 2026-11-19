"""Current-reference report consumers; all numeric inputs here are synthetic fixtures."""

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "third-party-test-kit")]
from compute_reference import target
from dataset_contract import save, sha
from reference_pair_statistics import arm, pair
from third_party_report import reference_card, render_reference_bundle


def fixture_result():
    result = pair(arm([10**12, 2*10**12], 1, .5), arm([10**12, 2*10**12], 1, .75))
    result.update(status="measured", full_model_certificate=True,
        certificate_kind="matched_complete_necessary_work_and_full_reference_lower_bound")
    return result


def test_current_report_card_never_substitutes_the_historical_percentage():
    historical = dict(formal_targets={"3.4.1": dict(display_value="59.7573%", accepted=True)})
    assert reference_card(fixture_result(), historical)[1] == "Incomplete"
    assert reference_card(dict(status="blocked", blockers=["missing native proof"]), historical)[1] == "Incomplete"
    assert reference_card(dict(status="not_requested"), historical)[1] == "59.7573%"


def sealed_reference(root, result=None):
    source = root / "compute-reference"
    source.mkdir()
    result = fixture_result() if result is None else result
    save(source / "status.json", result)
    for name in ("matched-reference-pair.json", "necessary-work-audit.json",
            "reference-graph-audits.json", "qualified-full-reference-statistics.json"):
        save(source / name, dict(fixture=True))
    save(source / "checksums.json", {p.name: sha(p) for p in source.iterdir() if p.is_file()})
    return result


def test_unqualified_reference_report_has_no_numeric_plot_or_historical_fallback(tmp_path):
    out = tmp_path / "report"
    value = dict(status="blocked", blockers=["missing <source> proof"])
    rendered = render_reference_bundle(tmp_path, out, value)
    assert rendered["plot"] is None and rendered["target"]["measured_value"] is None
    assert not (out / "current-reference-bounds.svg").exists()
    assert "missing &lt;source&gt; proof" in (out / "report.html").read_text()


def test_withdrawn_certificate_cannot_reach_numeric_plot_even_with_complete_flag(tmp_path):
    value = sealed_reference(tmp_path)
    (tmp_path / "compute-reference/matched-reference-pair.json").write_text("{}")
    out = tmp_path / "report"
    rendered = render_reference_bundle(tmp_path, out, value)
    assert rendered["plot"] is None
    assert rendered["target"]["measured_value"] is None
    assert not target(value)["accepted"]
    assert not (out / "current-reference-bounds.svg").exists()
    assert "撤回" in (out / "report.html").read_text()
    assert "59.7573" not in (out / "report.html").read_text()


def test_plotter_rejects_old_certificate_before_importing_matplotlib(tmp_path):
    from plot_compute_reference import draw

    with pytest.raises(ValueError, match="Only complete"):
        draw(fixture_result(), tmp_path)
    assert not (tmp_path / "compute-reference.svg").exists()


def test_complete_reference_reuse_rejects_changed_source_before_any_plot(tmp_path):
    from test_compute_reference_integration import qualified_bound

    value = sealed_reference(tmp_path, qualified_bound())
    (tmp_path / "compute-reference/qualified-full-reference-statistics.json").write_text("{}")
    with pytest.raises(ValueError, match="evidence changed"):
        render_reference_bundle(tmp_path, tmp_path / "report", value)
    assert not (tmp_path / "report").exists()


def test_complete_reference_reuse_draws_both_real_formats(tmp_path):
    from test_compute_reference_integration import qualified_bound

    matplotlib = pytest.importorskip("matplotlib")
    if matplotlib.__version__ != "3.10.8":
        pytest.skip("Renderer uses separately pinned matplotlib3.10.8")
    value = sealed_reference(tmp_path, qualified_bound())
    out = tmp_path / "report"
    result = render_reference_bundle(tmp_path, out, value)
    assert result["plot"] == "current-reference-bounds.svg"
    assert (out / "current-reference-bounds.png").read_bytes().startswith(b"\x89PNG")
    assert "75%" in (out / result["plot"]).read_text()
    assert target(value)["accepted"] is True
    assert "两份原始算量" in (out / "report.html").read_text()
