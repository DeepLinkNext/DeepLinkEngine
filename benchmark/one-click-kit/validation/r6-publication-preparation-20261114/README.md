# r6 publication preparation checkpoint

The active inference remains on execution commit
`9c4699be28425d3d05112cc31214e801e59cbf33`. MMLU B0 completed200/200 requests
with zero failures; B1 entered formal timing after the same126-request synthetic
warmup and the Torch/Triton measurement boundary. The full five-profile campaign
is still running. No new full-campaign pass or final-goal attainment is claimed.

`source-binding.json` pins the prepared benchmark exporter and website PRs.
The exporter accepts complete fresh/reused/mixed campaigns, checks each stage’s
physical cards and preserves original evidence. Figures use current provenance
and goal outcomes.16 publication/goal/stability/figure tests passed. The website
shows all five profiles, both goal tiers and quality/unresolved denominators;
7 public-projection browser cases passed. The website still explicitly selects
the prior published report until the new results are audited and appended.

`watch_publication_candidate.py CONFIG.json` is the optional local review helper.
It waits for the exact native controller PID/start-time identity to exit, requires
a completed receipt, verifies its frozen postprocessor source hashes, then runs
only export and plotting. It never launches inference, retries failed tasks,
accesses the network, publishes results or marks final review complete. A failed
native campaign or export produces an explicit failure receipt. The current
helper runs outside the sealed inference directory and checks every300seconds.
The one-click runner itself continues to generate its original sealed report
and plots. Final website publication and acceptance remain separate review steps.
