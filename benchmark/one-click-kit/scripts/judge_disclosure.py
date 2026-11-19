"""Describe the reference Engine's built-in auxiliary answer scoring."""

NOTICE_ZH = (
    "评分说明：若选择HLE/Frontier，脚本复用同一份Qwen3.5-35B-A3B权重，"
    "在单独的固定B1阶段为两臂答案辅助评分，无需另备模型或外部裁判服务。"
    "评分增加整轮耗时，但其流量不计solver吞吐、TTFT/TPOT或单位token成本。"
    "这是同模型辅助评分，不是独立模型验证；裁判未决保留在分母并单列，"
    "不等同引擎请求失败。MMLU200使用固定答案和规则评分。"
)
NOTICE_EN = (
    "Scoring: HLE/Frontier reuse the same Qwen3.5-35B-A3B weights in separate "
    "fixed-B1 stages to score both solver arms. No additional model or external "
    "judge service is required. Scoring adds campaign time; judge traffic is "
    "excluded from solver throughput, TTFT/TPOT and unit-token cost. This is "
    "same-model auxiliary scoring, not independent-model validation. Unresolved "
    "judgments stay in the planned denominator and are reported separately; "
    "they are not engine request failures. MMLU200 uses fixed-answer rule scoring."
)


def policy():
    return {
        "datasets": ["hle", "frontier"],
        "model": "Qwen3.5-35B-A3B",
        "weights_shared_with_solver": True,
        "service_role": "B1",
        "both_solver_arms_share_judge": True,
        "additional_model_or_external_service_required": False,
        "separate_inference_stages": True,
        "adds_campaign_time": True,
        "included_in_solver_performance_or_unit_cost": False,
        "independent_model_validation": False,
        "unresolved_is_engine_request_failure": False,
        "mmlu_scoring": "fixed_answer_rules",
        "notice_zh": NOTICE_ZH,
        "notice_en": NOTICE_EN,
    }
