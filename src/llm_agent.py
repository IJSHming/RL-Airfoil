"""LLM-Meta-Airfoil 元优化智能体模块：
基于 OpenAI 官方 SDK 与本地部署的大语言模型 (如 Qwen/Qwen3.8-27B-FP8)，
结合通用空气动力学物理原理与贝叶斯优化信赖域状态，
为 18 维 Kulfan (CST) TuRBO-1 提供停滞自适应元参数调节与宏观几何方向引导。

严格遵守科研诚信与零静默降级准则：
1. 严禁在提示词中硬编码最优解数值或指令性偏移区间；
2. 所有几何偏移通道采用对称边界，由模型根据空气动力学原理与实时反馈自主决策；
3. 严禁在 LLM 离线或解析失败时静默回退人工启发式字典。
"""

import json
import re
from typing import Any, Dict, List
import numpy as np

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


LLM_SYSTEM_PROMPT = """You are a senior aerodynamicist and Bayesian Optimization meta-controller.
Your goal is to maximize the penalized lift-to-drag ratio (L/D) of a 2D subsonic/transonic airfoil represented by an 18-dimensional Kulfan CST parameterization (8 upper Bernstein weights, 8 lower Bernstein weights, 1 leading-edge weight, 1 trailing-edge thickness) under industrial geometric and structural constraints.

General Aerodynamic & Geometric Constraints:
1. Geometric feasibility is strictly enforced: maximum relative thickness t_max in [8%, 20%], maximum thickness location x_tmax in [20%, 55%] chord, structural spar thickness minimums at 25% and 70% chord, non-zero leading-edge radius and trailing-edge angle, monotonic aft thickness taper (no fishtailing), smooth non-wavy upper surface, and pitching moment CM >= -0.25.
2. Lift-to-drag performance balances circulation (camber and aft loading), pressure-gradient control (suction-peak curvature and leading-edge radius), and profile drag / structural thickness limits.

Decision Mode:
- Decide Trust-Region Bayesian Optimization (TuRBO-1) hyperparameters adapted to D=18:
  * length_scale_factor in [0.70, 1.50]: temporary multiplier applied to the current trust-region side length L (>1.0 expands search to escape local plateaus; <1.0 tightens search for local refinement).
  * prob_perturb in [0.25, 1.00]: per-coordinate perturbation probability (baseline is min(20/18, 1.0) = 1.0).
  * exploration_weight in [0.70, 1.50]: Thompson sampling posterior standard-deviation multiplier (>1.0 boosts exploration; <1.0 favors exploitation).
- Optionally provide 'directional_strategy' specifying smooth macro CST shape perturbations along orthogonal physical modes, or set 'directional_strategy' to null when pure local GP sampling is preferable:
  * camber_shift in [-0.10, 0.10]: symmetric mid-chord camber mode (shifts both upper and lower surfaces equally with a smooth bell envelope).
  * aft_camber_shift in [-0.08, 0.08]: aft-chord camber / rear-loading mode (shifts both surfaces smoothly over the rear half of the chord).
  * thickness_shift in [-0.04, 0.04]: symmetric thickness mode (separates or compresses upper and lower surfaces smoothly).
  * le_radius_shift in [-0.03, 0.03]: leading-edge weight adjustment controlling nose bluntness.

Output ONLY a valid JSON object matching this exact schema:
{
  "thought": "1-2 sentence aerodynamic/BO rationale (under 45 words).",
  "length_scale_factor": float in [0.70, 1.50],
  "prob_perturb": float in [0.25, 1.00],
  "exploration_weight": float in [0.70, 1.50],
  "directional_strategy": {
    "camber_shift": float in [-0.10, 0.10],
    "aft_camber_shift": float in [-0.08, 0.08],
    "thickness_shift": float in [-0.04, 0.04],
    "le_radius_shift": float in [-0.03, 0.03]
  }
}
"""


class OptimizedLLMMetaAgent:
    """面向 2D 翼型宏观气动决策与信赖域调控的真实 LLM 元控制器智能体."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:30000/v1",
        model: str = "Qwen/Qwen3.8-27B-FP8",
        api_key: str = "EMPTY",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.endpoint = f"{self.base_url}/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.interaction_log: List[Dict[str, Any]] = []

        if HAS_OPENAI:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=1,
            )
        else:
            self.client = None

    def build_user_prompt(self, state: Dict[str, Any]) -> str:
        """根据当前优化状态与触发原因构建无泄漏客观状态提示词."""
        step = state["step"]
        best_cl = state.get("best_cl", 0.0)
        best_cd = state.get("best_cd", 0.0)
        best_cm = state.get("best_cm", 0.0)
        best_ld = state.get("best_ld", 0.0)
        best_t_max = state.get("best_t_max", 0.12)
        stag = state.get("stagnation_count", 0)
        trigger_reason = state.get("trigger_reason", "stagnation")

        if trigger_reason == "post_init":
            phase_str = (
                "POST-INIT DIAGNOSTIC (Initial 15 local samples around baseline just finished; no stagnation event yet): "
                "Assess initial aerodynamic state and decide initial trust-region hyperparameters and optional macro shape direction."
            )
        elif step <= 95:
            phase_str = (
                f"EARLY/MID STAGNATION INTERVENTION (Stagnation = {stag} steps, Best L/D = {best_ld:.2f}): "
                "Local GP sampling has stalled. Adapt trust-region exploration hyperparameters and optionally propose a smooth macro shape shift."
            )
        else:
            phase_str = (
                f"LATE STAGNATION INTERVENTION (Stagnation = {stag} steps, Best L/D = {best_ld:.2f}): "
                "Local refinement has stalled. Adapt trust-region parameters and optionally refine shape within geometric constraints."
            )

        prompt = f"""=== Current 2D Airfoil Optimization State ===
- Phase: {phase_str}
- Baseline Airfoil: {state.get('airfoil_name', 'NACA0012')} | Re = {state['reynolds']:.2e} | AoA = {state['alpha']} deg
- Progress: Step {step} / {state['max_evals']} (Remaining Budget: {state['budget_left']})
- Initial L/D: {state['init_ld']:.2f} | Current Best Feasible L/D: {best_ld:.2f}
- Current Best Aerodynamics & Geometry: CL = {best_cl:.4f}, CD = {best_cd:.5f}, CM = {best_cm:.4f} (Limit >= -0.25), t_max = {best_t_max*100:.2f}% (Allowed: [8%, 20%])
- TuRBO Status: Effective TR Length L = {state['tr_length']:.4f}, Stagnation = {stag}, Successes = {state['success_counter']}/{state['success_tolerance']}

Concisely decide meta-parameters and optional directional strategy. Output final JSON:"""
        return prompt

    def decide_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """向本地部署的真实 LLM 请求下一阶段超参策略与宏观几何方向 (严禁静默降级)."""
        if not HAS_OPENAI or self.client is None:
            raise RuntimeError(
                "[Research Integrity Violation] openai client is not initialized or not installed! "
                "Silent degradation to hand-crafted heuristic dictionary is strictly forbidden."
            )

        user_prompt = self.build_user_prompt(state)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": LLM_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.6,
                top_p=0.90,
                max_tokens=1024,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            raw_text = response.choices[0].message.content or ""
            clean = re.sub(r"```json\s*", "", raw_text)
            clean = re.sub(r"```\s*", "", clean).strip()
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                scale_L = float(np.clip(data.get("length_scale_factor", 1.0), 0.70, 1.50))
                prob_p = float(np.clip(data.get("prob_perturb", 1.0), 0.25, 1.00))
                exp_w = float(np.clip(data.get("exploration_weight", 1.0), 0.70, 1.50))
                strat = data.get("directional_strategy")

                action: Dict[str, Any] = {
                    "thought": str(data.get("thought", "")),
                    "length_scale_factor": scale_L,
                    "prob_perturb": prob_p,
                    "exploration_weight": exp_w,
                    "directional_strategy": None,
                }
                if isinstance(strat, dict):
                    action["directional_strategy"] = {
                        "camber_shift": float(np.clip(strat.get("camber_shift", 0.0), -0.10, 0.10)),
                        "aft_camber_shift": float(np.clip(strat.get("aft_camber_shift", 0.0), -0.08, 0.08)),
                        "thickness_shift": float(np.clip(strat.get("thickness_shift", 0.0), -0.04, 0.04)),
                        "le_radius_shift": float(np.clip(strat.get("le_radius_shift", 0.0), -0.03, 0.03)),
                    }
                self.interaction_log.append({
                    "step": state.get("step"),
                    "trigger_reason": state.get("trigger_reason", "stagnation"),
                    "user_prompt": user_prompt,
                    "raw_response": raw_text,
                    "action": action,
                })
                return action
            raise ValueError(f"Failed to parse valid JSON from LLM response:\n{raw_text}")
        except Exception as e:
            raise RuntimeError(
                f"[LLM Fatal Error] Real LLM request to {self.base_url} failed with {type(e).__name__}: {e}. "
                "Silent degradation to hand-crafted heuristic dictionary is strictly forbidden."
            ) from e
