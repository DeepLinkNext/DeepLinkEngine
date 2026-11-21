"""Check the shared download index while preserving each dependency's release and scope."""
import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def build_index(runtime, evaluation, source_build=None):
    # Existing installer callers use the same aggregate index, while their
    # kind filters continue to select only the fixed runtime artifacts.
    if source_build is None:
        catalog = ROOT / "engine/official-build-dependencies.json"
        if catalog.is_file():
            source_build = json.loads(catalog.read_bytes())
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
    if source_build is not None:
        if source_build.get("schema") != "deeplinkengine-source-build-dependency-catalog/v1":
            raise ValueError("Unknown source-build dependency catalog")
        for record in source_build["artifacts"]:
            name = record["name"]
            if (record.get("kind") != "source-build" or not record.get("applies_to")
                    or not isinstance(record["bytes"], int) or record["bytes"] <= 0
                    or re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is None
                    or re.fullmatch(r"[0-9a-f]{64}", record["provenance_sha256"]) is None):
                raise ValueError("Incomplete source-build dependency identity")
            for key in ("url", "api_url", "upstream_url", "provenance_url"):
                link = urlsplit(record[key])
                if link.scheme != "https" or not link.hostname or link.username or link.password or link.fragment:
                    raise ValueError("Invalid source-build dependency URL: " + key)
            for key in ("url", "upstream_url"):
                if unquote(urlsplit(record[key]).path.rsplit("/", 1)[-1]) != name:
                    raise ValueError("Source-build dependency filename differs from its URL")
            if not record["provenance_url"].startswith(prefix):
                raise ValueError("Source-build dependency provenance must use the central release")
            rows.append(dict(record))
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
                           json.loads((ROOT / "benchmark/one-click-lock.json").read_bytes()),
                           json.loads((ROOT / "engine/official-build-dependencies.json").read_bytes()))
    path = ROOT / "engine/dependencies.json"
    if args.write:
        path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n")
    elif json.loads(path.read_bytes()) != expected:
        raise ValueError("Dependency index differs from the installation lock files")
    print("PASS: " + str(len(expected["artifacts"])) + " artifacts share one download index with explicit dependency scopes")


if __name__ == "__main__":
    main()
