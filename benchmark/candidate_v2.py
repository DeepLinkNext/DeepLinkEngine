"""Evaluate the frozen V2 criteria at the level supported by candidate archives."""

from math import isclose

from benchmark.full_report import (
    COUNTS, MMLU_UNIT_COST_REDUCTION_MIN, PERFORMANCE_LIMITS, QUALITY_TRACKS,
    context_decisions, cost_decisions, cost_per_million, coverage, decimal_text, number, performance_clears, performance_ratios,
    published_rental, quality_gates, reduction,
)


ROLES = {"B0": "B0", "B1": "candidate"}
RAW_FIXED_FIVE = (
    "两臂的冻结执行计划、运行回执和逐请求原件（含封存清单）",
    "MMLU-Pro 逐题答案核验及 HLE-Verified、FrontierScience 逐题裁判原件",
)


def verdict(value, reason):
    label = {True: "met", False: "not_met", None: "missing_evidence"}
    return {"status": label[value], "met": value, "reason": reason}


def recover_priority_tracks(archive, source):
    """Recover the native RMS track totals from same-judge paired transitions."""
    priority = source.get("priority_pair", {})
    bound = archive["sealed_reports"]["priority"]
    if (
        source.get("schema") != "vllm-hust-r10-native-rms-evidence/v1"
        or source.get("baseline_id") != archive["baseline_id"]
        or priority.get("release") != archive["release"]
        or priority.get("aggregate_sha256") != bound["seal_sha256"]
        or priority.get("files_verified") != bound["files_verified"]
        or priority.get("judges") != archive["judges"]
    ):
        raise ValueError("Native RMS track summary differs from the archived judge run")

    recovered = {}
    for name, denominators in QUALITY_TRACKS.items():
        scope = priority.get("scope", {}).get(name, {})
        early_roles = {"B0": "B0", "B1": "r10_corrected"}
        for role, arm in ROLES.items():
            earlier = scope.get(early_roles[role], {})
            later = archive["measurements"][name][arm]
            if any(earlier.get(key) != later[key] for key in ("source_seal_sha256", "requests", "correct", "unresolved")):
                raise ValueError(f"{name} {role} track evidence belongs to a different solver or grade tally")
        tracks = {role: {track: {"total": 0, "correct": 0, "unresolved": 0}
                         for track in denominators} for role in ROLES}
        transitions = source["diagnostics"][name]["B0_vs_fixed"]["transitions"]
        tally_status = ("correct", "wrong", "unresolved")
        for key, count in transitions.items():
            track, left, right = key.split("/")
            if track not in denominators or left not in tally_status or right not in tally_status or type(count) is not int or count < 0:
                raise ValueError(f"Invalid paired {name} grade transition")
            for role, state in (("B0", left), ("B1", right)):
                row = tracks[role][track]
                row["total"] += count
                if state != "wrong":
                    row[state] += count
        recovered[name] = tracks
    return recovered


def valid_tracks(tracks, denominators, arm):
    return (
        set(tracks) == set(denominators)
        and all(
            tracks[name].get("total") == count
            and all(type(tracks[name].get(key)) is int and tracks[name][key] >= 0 for key in ("correct", "unresolved"))
            and tracks[name]["correct"] + tracks[name]["unresolved"] <= count
            for name, count in denominators.items()
        )
        and all(sum(row[key] for row in tracks.values()) == arm.get(key) for key in ("correct", "unresolved"))
    )


def valid_judge(judge, arm, count):
    unresolved = arm.get("unresolved")
    return (
        type(unresolved) is int and 0 <= unresolved <= count
        and judge.get("planned") == judge.get("recorded") == count
        and judge.get("failed") == judge.get("unreturned") == 0
        and judge.get("diagnostic") is False
        and judge.get("unresolved_grades") == unresolved
        and judge.get("successful") == count - unresolved
        and arm.get("judge_invalid") == unresolved
    )


def grade_details(archive, recovered):
    inputs = {"current_mmlu_pair": {"arms": {}}, "current_dataset_quality": {}}
    proof = {}
    mmlu = archive["measurements"]["mmlu"]
    for role, arm in ROLES.items():
        observed = mmlu[arm]
        if observed["requests"] == COUNTS["mmlu"] and observed["failed"] == observed.get("unresolved") == 0:
            inputs["current_mmlu_pair"]["arms"][role] = {
                "quality": {"planned_denominator": COUNTS["mmlu"], "recorded": observed["requests"],
                            "unreturned": 0, "correct": observed["correct"]}}
    for name, denominators in QUALITY_TRACKS.items():
        proof[name] = {"source": "paired_grade_transitions" if name in recovered else "candidate_summary", "arms": {}}
        for role, arm in ROLES.items():
            observed = archive["measurements"][name][arm]
            tracks = recovered[name][role] if name in recovered else observed.get("track_scores", {})
            judge = archive["judges"].get(f"{name}-judge-{role}", {})
            if not (valid_tracks(tracks, denominators, observed) and valid_judge(judge, observed, COUNTS[name])):
                continue
            inputs["current_dataset_quality"][f"{name}-{role}"] = {
                "tracks": {key: {"total": scores["total"], "successes": scores["correct"]}
                           for key, scores in tracks.items()}}
            proof[name]["arms"][role] = tracks
    return quality_gates(inputs), proof


def validate_original(archive, original, proof):
    if (original["release"], original["baseline_id"]) != (archive["release"], archive["baseline_id"]):
        raise ValueError("r10 original audit belongs to another candidate or baseline")
    for name in COUNTS:
        for role, arm in ROLES.items():
            expected = archive["measurements"][name][arm]
            row = original["performance"][name][role]
            if (row["seal_sha256"] != expected["source_seal_sha256"]
                    or row["verified_files"] <= 0
                    or any(row[field] != expected[field] for field in ("requests", "failed", "output_tokens"))
                    or any(not isclose(row[field], expected[field], rel_tol=1e-8)
                           for field in ("duration_seconds", "output_tokens_per_second", "mean_ttft_ms", "mean_tpot_ms"))):
                raise ValueError(f"r10 original performance audit differs for {name} {role}")
    for role, arm in ROLES.items():
        row = original["mmlu"][role]
        expected = archive["measurements"]["mmlu"][arm]
        if (row["seal_sha256"] != expected["source_seal_sha256"]
                or row["requests"] != expected["requests"] or row["correct"] != expected["correct"]):
            raise ValueError(f"r10 MMLU-Pro answer tally differs for {role}")
        for name in QUALITY_TRACKS:
            audited = original["quality"][name][role]["tracks"]
            expected_tracks = proof[name]["arms"].get(role, {})
            if (set(audited) != set(expected_tracks)
                    or any(audited[track] != {key: expected_tracks[track][key] for key in ("total", "correct", "unresolved")}
                           for track in audited)):
                raise ValueError(f"r10 original grades differ for {name} {role}")


def assess_speed(archive, quality, proof, original=None):
    datasets = {}
    for name, count in COUNTS.items():
        arms = {role: archive["measurements"][name][arm] for role, arm in ROLES.items()}
        ratios = performance_ratios(arms)
        completed = all(row["requests"] == count and row["failed"] == 0 for row in arms.values())
        gate = quality[name]
        trace = name in ("swe", "terminal")
        quality_met = gate["met"] if trace or "drop_pp" in gate else None
        if trace:
            quality_reason = "固定公开轨迹仅核验推理服务性能。"
        else:
            quality_reason = {
                "mmlu": "需要 MMLU-Pro 固定 200 题两臂的完整答案统计与正确数。",
                "hle": "需要两臂 gold 74 / revision 126 分层分数及同轮裁判回执统计。",
                "frontier": "需要两臂 Olympiad 100 / Research 60 赛道分数及同轮裁判回执统计。",
            }[name] if quality_met is None else gate["reason"]
        missing_arms = [role for role in ROLES if role not in proof[name]["arms"]] if name in proof else []
        if quality_met is None and missing_arms:
            quality_reason += " 缺项运行臂：" + "、".join(missing_arms) + "。"
        decisions = {}
        for level, limits in PERFORMANCE_LIMITS.items():
            performance_met = performance_clears(ratios, *limits) if all(value is not None for value in ratios.values()) else None
            if completed and (performance_met is False or quality_met is False):
                reasons = []
                if performance_met is False:
                    reasons.append(f"性能未同时满足吞吐≥{limits[0]}×及平均 TTFT/TPOT 各≤{limits[1]}×。")
                if quality_met is False:
                    reasons.append("正确率下降超过 V2 门槛。")
                decisions[level] = verdict(False, "".join(reasons))
            elif not completed or performance_met is None or quality_met is None:
                gaps = []
                if not completed:
                    gaps.append("缺固定请求数或零失败记录。")
                if performance_met is None:
                    gaps.append("缺吞吐或平均 TTFT/TPOT 原值。")
                if quality_met is None:
                    gaps.append(quality_reason)
                decisions[level] = verdict(None, "".join(gaps))
            else:
                decisions[level] = verdict(True, "本数据集满足完整请求、适用的质量门槛及三项性能门槛。")
        datasets[name] = {
            "ratios": {key: decimal_text(value) for key, value in ratios.items()},
            "completed_in_summary": completed,
            "quality": {**verdict(quality_met, quality_reason), "drop_pp": gate.get("drop_pp"),
                        **proof.get(name, {})},
            **decisions,
        }
    stage = {"evidence_scope": "sealed_summary", "datasets": datasets,
             "raw_reaudit": verdict(None, "独立逐请求复核需要以下原件。"), "raw_missing": list(RAW_FIXED_FIVE)}
    if original is not None:
        stage.update(
            evidence_scope="cpu_audited_originals",
            raw_reaudit=verdict(True, "已按原件封存复算两臂固定五项的吞吐和平均 TTFT/TPOT、MMLU-Pro 逐题答案统计，以及 HLE-Verified 和 FrontierScience 的逐题裁判统计。"),
            raw_missing=[],
            available="两臂固定五项逐请求记录、运行回执、封存清单和对应判分已完成 CPU 复核。",
            original_evidence=original,
        )
    for level in PERFORMANCE_LIMITS:
        winners = [name for name, row in datasets.items() if row[level]["met"] is True]
        if winners:
            stage[level] = {**verdict(True, "同一数据集同时达到所需门槛。"), "datasets": winners}
        elif any(row[level]["met"] is None for row in datasets.values()):
            stage[level] = verdict(None, "尚需补齐相关数据集的性能或质量证据。")
        else:
            stage[level] = verdict(False, "各数据集均未同时达到所需门槛。")
    return stage


def matched_utilization(observation, meter):
    samples = observation["utilization"]
    if (observation["meter"] == meter and coverage(samples)
            and len(samples.get("per_card_percent", {})) == meter["occupied_cards"]
            and number(samples["covered_seconds"]) == number(meter["window_seconds"])):
        return samples
    return None


def assess_cost(archive, accelerator_count, speed, fixed_b0, original=None):
    rental = published_rental()
    rate = rental["monthly_cost_per_card_cny"]
    arms = {}
    for role, arm in ROLES.items():
        observed = archive["measurements"]["mmlu"][arm]
        arms[role] = {
            "occupied_cards": accelerator_count, "window_seconds": str(observed["duration_seconds"]),
            "successful_output_tokens": observed["output_tokens"],
            "measurement_seal_sha256": observed["source_seal_sha256"],
        }
    costs = cost_per_million(arms, dict.fromkeys(ROLES, rate))
    savings = reduction(costs)
    cost_met = savings >= MMLU_UNIT_COST_REDUCTION_MIN
    mmlu = speed["datasets"]["mmlu"]
    baseline_util = matched_utilization(fixed_b0["mmlu"], arms["B0"])
    candidate_util = matched_utilization(original["mmlu"]["B1"], arms["B1"]) if original is not None else None
    gain = (number(candidate_util["mean_percent"]) - number(baseline_util["mean_percent"])
            if baseline_util is not None and candidate_util is not None else None)
    if not cost_met:
        milestone = verdict(False, "按公开租金折算的 MMLU-Pro 200 单位成本下降未达到门槛。")
    elif mmlu["quality"]["met"] is False:
        milestone = verdict(False, "MMLU-Pro 200 的质量门槛未达成。")
    elif gain is not None and mmlu["quality"]["met"] is True and mmlu["completed_in_summary"]:
        milestone = verdict(gain >= 10, "MMLU-Pro 200 完整回答满足质量和服务要求。按公开月租折算的单位成本下降≥10%，双卡 AI Core 均值须提高≥10个百分点。")
    else:
        gaps = []
        if baseline_util is None:
            gaps.append("缺固定 B0 同窗逐卡 AI Core 采样。")
        if candidate_util is None:
            gaps.append("缺所选候选同窗逐卡 AI Core 采样。")
        if mmlu["quality"]["met"] is not True or not mmlu["completed_in_summary"]:
            gaps.append("缺 MMLU-Pro 200 完整回答的质量或服务完成证据。")
        milestone = verdict(None, "".join(gaps))
    tenant = fixed_b0["four_tenant"]
    tenant_meter = tenant["meter"]
    baseline_tenant_cost = cost_per_million({"B0": tenant_meter}, {"B0": rate})["B0"]
    return {
        "cost_basis": "published_rental", "published_rental": rental,
        "mmlu_full_answers": {
            "arms": {role: {**arm, "cost_per_million_output_tokens_cny": decimal_text(costs[role])}
                     for role, arm in arms.items()},
            "unit_cost_reduction_percent": decimal_text(savings),
            "cost_gate": verdict(cost_met, f"按公开月租折算，单位成本下降≥{MMLU_UNIT_COST_REDUCTION_MIN}%。"),
            "utilization": {"B0": baseline_util, "B1": candidate_util},
            "same_window_utilization_gain_pp": decimal_text(gain),
        },
        "four_tenant": {
            "arms": {"B0": {**tenant_meter, "cost_per_million_output_tokens_cny": decimal_text(baseline_tenant_cost)},
                     "B1": None},
            "utilization": {"B0": matched_utilization(tenant, tenant_meter), "B1": None},
            "unit_cost_reduction_percent": None,
        },
        "available": ("MMLU-Pro 200 完整回答的两臂占卡耗时、输出量和公开租金。"
                      + ("两臂同窗逐卡 AI Core 采样已复算。" if gain is not None else "")
                      + "固定 B0 已有真实 4K 四租户计量。"),
        "milestone": milestone,
        "final": verdict(None, "缺所选候选在相同真实输入集上的 4K 四租户 C24 运行、逐租户回执和同窗双卡采样。"),
    }


def attach_context(indicators, observed, archive, fixed_b0):
    baseline = fixed_b0["context"]
    paired = observed["paired"]
    if (observed.get("schema") != "deeplinkengine-r10-v2-context-projection/v1"
            or observed["release"] != archive["release"]
            or observed["input_sha256"] != baseline["input_sha256"]
            or paired["arm_seals"]["B0"] != baseline["seal_sha256"]
            or observed["datasets"] != baseline["workload"]["datasets"]
            or [row["phase"] for row in paired["comparison"]] != baseline["phases"]):
        raise ValueError("Real-context evidence differs from the archived candidate or frozen B0 contract")
    context = context_decisions(paired, observed["audit"])
    indicators["4.3"].update(
        available="固定 B0 和所选候选均已有 Terminal-Bench 2.1 真实 4K、16K、32K 与四租户 C24 封存。",
        milestone=verdict(context["milestone"]["met"], context["milestone"]["reason"]),
        final=verdict(context["final"]["met"], context["final"]["reason"]),
        paired=paired,
        physical_cards=observed["physical_cards"],
        candidate_runtime=observed["candidate_runtime"],
        measured_ratios=context["measured_ratios"],
        measured_window_ratios=context["measured_window_ratios"],
        blocking_errors=context["blocking_errors"],
    )
    costs = cost_decisions({}, {}, paired, observed["audit"], None)
    tenant = costs["formal_cost"]["four_tenant"]
    indicators["4.4"].update(
        four_tenant={"arms": tenant["arms"], "utilization": tenant["utilization"],
                     "unit_cost_reduction_percent": costs["four_tenant_cost_reduction_percent"],
                     "ai_core_mean_percent": costs["four_tenant_ai_core_mean_percent"]},
        final=verdict(costs["final"]["met"], costs["final"]["reason"]),
    )


def assess(archive, selection, fixed_b0, priority_source=None, original=None, context=None):
    recovered = recover_priority_tracks(archive, priority_source) if priority_source is not None else {}
    quality, proof = grade_details(archive, recovered)
    if original is not None:
        validate_original(archive, original, proof)
    speed = assess_speed(archive, quality, proof, original)
    ctx = fixed_b0["context"]
    if fixed_b0.get("schema") != "deeplinkengine-fixed-b0-v2-projection/v1" or fixed_b0.get("baseline_id") != archive["baseline_id"]:
        raise ValueError("Fixed B0 V2 evidence does not match the archived baseline")
    indicators = {
        "4.1": {
            "available": "已有固定 B0 的 MMLU-Pro 200 / out256 算力认证，以及此候选的固定五项吞吐和时延。",
            "baseline_reference": fixed_b0["compute"],
            "milestone": verdict(None, "缺候选 MMLU-Pro 固定200/out256 三轮原生双卡同步计时、必要算量和同工况参考分母认证，无法核验效率下界≥50%。"),
            "final": verdict(None, "同一候选算力认证未附在封存汇总中，无法核验效率下界≥90%。"),
        },
        "4.2": speed,
        "4.3": {
            "available": ("已有两臂 4K / C8 和 16K / C8 合成闭环结果。固定 B0 另有真实 "
                          + "、".join(ctx["workload"]["datasets"]) + " 4K、16K、32K 和四租户封存。"),
            "baseline_context": ctx,
            "milestone": verdict(None, "缺所选候选在相同真实输入集上的 4K、16K 配对原件、完整日志和同窗采样。"),
            "final": verdict(None, "缺所选候选的真实 32K 原件、五个 60 秒窗口的吞吐与尾时延，以及日志与采样覆盖。"),
        },
        "4.4": assess_cost(archive, selection["accelerator_count"], speed, fixed_b0, original),
    }
    if context is not None:
        attach_context(indicators, context, archive, fixed_b0)
    for row in indicators.values():
        milestone, final = row["milestone"], row["final"]
        states = {milestone["status"], final["status"]}
        row["status"] = milestone["status"] if len(states) == 1 else "mixed"
        row["reason"] = "里程碑 " + milestone["reason"] + " 结题 " + final["reason"]
    return indicators
