"""Plot sealed current reference statistics after all model services exit."""

import argparse
import json
from pathlib import Path

from dataset_contract import sha


def draw(result, out):
    from compute_reference import target

    if (result.get("status") != "measured" or result.get("full_model_certificate") is not True
            or target(result)["status"] != "current_workload_measured"):
        raise ValueError("Only complete current reference results can be plotted")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from compute_reference import bound_display

    roles = ("B0", "B1")
    colors = ("#426182", "#137d66")
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), layout="constrained")
    bounds = [float(result["arms"][r]["efficiency_lower_percent"]) for r in roles]
    axes[0].bar(roles, bounds, color=colors, width=.5)
    for index, value in enumerate(bounds):
        axes[0].text(index, value, "≥ " + bound_display(result["arms"][roles[index]]["efficiency_lower_percent"]) + "%", ha="center", va="bottom")
    for value, label in ((50, "Milestone50%"), (90, "Final90%")):
        axes[0].axhline(value, color="#526170", linestyle="--", linewidth=1, label=label)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -.13), ncol=2, fontsize=9, frameon=False)
    axes[0].set_ylim(0, max(110, max(bounds) * 1.18))
    axes[0].set_ylabel("Reference efficiency lower bound (%)")
    axes[0].set_title("Matched F/T and independent reference", fontsize=11)
    for index, role in enumerate(roles):
        arm = result["arms"][role]
        low, high = map(float, arm["achieved_tflops_per_card_interval"])
        upper = float(arm["reference_tflops_per_card_upper"])
        axes[1].plot([index, index], [low, high], color=colors[index], linewidth=8, solid_capstyle="round")
        axes[1].scatter([index], [upper], marker="_", s=350, linewidths=2, color=colors[index])
        axes[1].annotate("achieved[" + bound_display(arm["achieved_tflops_per_card_interval"][0]) + ", " + bound_display(arm["achieved_tflops_per_card_interval"][1], upper=True) + "]", (index, low), xytext=(6, -14), textcoords="offset points", fontsize=8)
        axes[1].annotate("reference ≤ " + bound_display(arm["reference_tflops_per_card_upper"], upper=True), (index, upper), xytext=(6, 5), textcoords="offset points", fontsize=8)
    axes[1].set_xticks((0, 1), roles)
    axes[1].set_xlim(-.4, 1.6)
    axes[1].set_ylim(bottom=0)
    axes[1].set_ylabel("TFLOP/s/card")
    axes[1].set_title("Compute interval and reference upper bound", fontsize=11)
    capture = result.get("capture_profile", {"max_tokens":8192, "formal_repetitions":1})
    fig.suptitle(f"MMLU200 · out{capture['max_tokens']} · {capture['formal_repetitions']} repetitions/arm · BF16 TP2", fontsize=12)
    out = Path(out)
    for extension in ("svg", "png"):
        kwargs = {"metadata": {"Date": None}} if extension == "svg" else {"dpi": 170}
        fig.savefig(out / ("compute-reference." + extension), **kwargs)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    seal = json.loads((args.run / "checksums.json").read_bytes())
    for name in ("status.json", "matched-reference-pair.json"):
        if sha(args.run / name) != seal.get(name):
            raise ValueError("Reference plot source is unsealed or changed: " + name)
    result = json.loads((args.run / "status.json").read_bytes())
    paired = json.loads((args.run / "matched-reference-pair.json").read_bytes())
    if any(result.get(key) != value for key, value in paired.items()):
        raise ValueError("Reference status and paired statistics differ")
    draw(result, args.out)


if __name__ == "__main__":
    main()
