"""Check that all fixed installation inputs share one release and one download index."""
import argparse
import json
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def build_index(runtime, evaluation):
    hub = runtime["dependency_release"]
    tag = hub["tag"]
    base = "https://github.com/DeepLinkNext/DeepLinkEngine/releases/"
    if hub["url"] != base + "tag/" + tag or hub["find_links_url"] != base + "expanded_assets/" + tag:
        raise ValueError("Dependency release and pip index differ")
    prefix = base + "download/" + tag + "/"
    if set(runtime["environment_sources"]) != set(runtime["required_python_packages"]):
        raise ValueError("Fixed Python dependencies are missing from the central index")
    rows = []

    def add(kind, record, name, url, version=None):
        if not url.startswith(prefix) or unquote(url.rsplit("/", 1)[-1]) != name:
            raise ValueError("Artifact is outside the declared dependency release: " + name)
        row = dict(kind=kind, name=name, bytes=record["bytes"], sha256=record["sha256"],
                   url=url, api_url=record["api_url"])
        if version:
            row["version"] = version
        if record.get("upstream_url"):
            row["upstream_url"] = record["upstream_url"]
        elif record.get("url") and record["url"] != url:
            row["upstream_url"] = record["url"]
        if record.get("provenance_url"):
            row["provenance_url"] = record["provenance_url"]
        rows.append(row)

    for record in runtime["assets"]:
        if "deepengine_url" in record:
            add("engine", record, record["name"], record["deepengine_url"])
    record = runtime["evaluator"]
    add("frozen-evaluator", record, record["name"], record["deepengine_url"])
    for package, record in runtime["environment_sources"].items():
        add("python", record, record["filename"], record["url"], runtime["required_python_packages"][package])
    for package, record in runtime.get("runtime_compatibility_sources", {}).items():
        add("python-compatibility", record, record["filename"], record["url"], record["version"])
    for key in ("tools", "reference_evidence"):
        record = evaluation[key]
        add(key, record, record.get("name", record.get("archive")), record["url"])
    names = [x["name"] for x in rows]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate dependency filenames")
    return dict(schema="deeplinkengine-dependency-index/v1", release=hub,
                artifacts=sorted(rows, key=lambda row: (row["kind"], row["name"])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="Regenerate the index after updating the lock files")
    args = parser.parse_args()
    expected = build_index(json.loads((ROOT / "engine/r8-lock.json").read_bytes()),
                           json.loads((ROOT / "benchmark/one-click-lock.json").read_bytes()))
    path = ROOT / "engine/dependencies.json"
    if args.write:
        path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n")
    elif json.loads(path.read_bytes()) != expected:
        raise ValueError("Dependency index differs from the installation lock files")
    print("PASS: " + str(len(expected["artifacts"])) + " fixed artifacts share one dependency release and download index")


if __name__ == "__main__":
    main()
