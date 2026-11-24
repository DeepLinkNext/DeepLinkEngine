"""Prepare a single-worker site from a DeepLinkEngine installation and the original B0 image."""
import argparse
import json
from pathlib import Path
import sys

from benchmark import baseline_bundle, one_click
from deeplink import installation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installation", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--physical-devices", required=True)
    parser.add_argument("--devices", required=True, help="Logical device IDs in the engine process")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New site JSON file")
    parser.add_argument("--ride-config", type=Path, help="Site RIDE endpoint and controller specs")
    args = parser.parse_args()
    cards = [int(x) for x in args.physical_devices.split(",")]
    if len(set(cards)) != 2 or len(cards) != 2 or min(cards) < 0:
        parser.error("Select two distinct physical NPUs")
    if args.out.exists():
        parser.error("Site output already exists")
    model = str(args.model.resolve(strict=True))
    root, receipt = installation(args.installation)
    one_click.verify_kit()
    kit = one_click.KIT / "scripts"
    sys.path.insert(0, str(kit))
    cache = args.cache.resolve()
    baseline = baseline_bundle.ensure_deployment(cache)
    if baseline is None:
        parser.error("Set DEEPLINKENGINE_B0_RUNTIME_CACHE to a filesystem with room to restore original018")
    endpoint = "http://127.0.0.1:18390"
    reference = dict(release="original018-cann8.5.1", endpoint=endpoint, argv=[
        receipt["python"], str(kit / "baseline_entry.py"), "--kind", "joint", "--phase", "serve",
        "--deployment", str(root / "sdk"), "--baseline-deployment", str(baseline), "--model", model,
        "--devices", args.devices, "--cache", "{output}/compiler", "--out", "{output}/service"])
    optimized = dict(release=receipt["release"], endpoint=endpoint, argv=[
        sys.executable, str(one_click.ROOT / "deeplink.py"), "serve", "--installation", str(root),
        "--model", model, "--physical-devices", "{cards}", "--devices", args.devices, "--port", "18390",
        "--served-model-name", "qwen3.5-35b", "--work-dir", "{output}/service"])
    label = "b1-" + receipt["release"].rsplit("-", 1)[-1]
    site = dict(physical_cards=cards, data_cache=str(cache), results=str(args.results.resolve()),
                engines={"b0-original018": reference, label: optimized})
    if args.ride_config:
        site["ride"] = json.loads(args.ride_config.read_bytes())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        json.dump(site, output, indent=2)
        output.write("\n")
    print(f"Site: {args.out}  Engines: b0-original018, {label}")


if __name__ == "__main__":
    main()
