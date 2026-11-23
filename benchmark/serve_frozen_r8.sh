#!/usr/bin/env bash
# Foreground adapter for the RIDE R8 program-cache experiment payload.
# Arguments: prepared launcher directory, verified runtime, new work directory, lease end (Unix epoch).
set -euo pipefail
umask 077
launcher=$(realpath "${1:?prepared RIDE launcher directory required}")
runtime=$(realpath "${2:?prepared R8 runtime required}")
work=${3:?new work directory required}
deadline=${4:?service lease end in Unix seconds required}
: "${VLLM_HUST_PHYSICAL_DEVICES:?physical devices must be assigned by the benchmark}"
[[ "$deadline" =~ ^[0-9]+$ && "$deadline" -gt "$(date +%s)" ]]
mkdir "$work"
cd "$work"
work=$(pwd -P)
mkdir launch-v2 tools
cp "$launcher/SHA256SUMS" "$launcher/issue6_r8_worker.sh" .
cp "$launcher/launch-v2/"{launch_overlay,core_capture,engine_capture_plugin}.py launch-v2/
cp "$launcher/tools/npu-smi" tools/
printf '%s  SHA256SUMS\n' 2acb388704dea97fe914da1e79046f6ae93165997473219a30de68579124a351 | sha256sum --check
sha256sum --check SHA256SUMS
bash issue6_r8_worker.sh "$work" reuse-runtime "$runtime"
bash issue6_r8_worker.sh "$work" native-preflight "$VLLM_HUST_PHYSICAL_DEVICES"
exec bash issue6_r8_worker.sh "$work" serve "$VLLM_HUST_PHYSICAL_DEVICES" "$deadline"
