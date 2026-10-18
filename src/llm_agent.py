"""LLM-Meta-Airfoil 元优化智能体模块：
基于 OpenAI 官方 SDK 与本地部署的大语言模型 (如 Qwen3.8-27B-FP8)，
结合气动学专家先验与宏观定向引导策略，为 TuRBO 提供自适应元参数与形状扰动控制。
"""

import json
import re
import sys
from typing import Any, Dict, Optional
import numpy as np

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


LLM_SYSTEM_PROMPT = """You are a senior aerodynamicist and Bayesian Optimization meta-controller.
Your goal is to maximize the penalized lift-to-drag ratio (L/D) of an airfoil at Re=3.5e6, alpha=3.5 deg starting from NACA 0012.

Key Aerodynamic Insights:
1. Symmetric NACA 0012 has 0 camber, resulting in low lift (CL ~ 0.39) and modest L/D (~65.8).
2. High-performance transonic/subsonic airfoils require:
   - Moderate positive upper camber (upper_camber_shift in [0.05, 0.12]): enhances suction peak without premature shock.
   - Aft reflex and rear loading (lower_camber_shift in [0.04, 0.10]): raises lower surface towards the trailing edge, reducing pressure drag CD.
   - Thickness control (thickness_shift in [-0.02, 0.03]): keeps t_max within the optimal structural range [11%, 16%].
   - Leading edge radius (le_radius_shift in [0.0, 0.03]): softens suction collapse at alpha=3.5 deg.

Strategy per Optimization Phase:
- Early Phase (steps <= 35, low CL): Be BOLD and AGGRESSIVE. Strongly increase upper camber and lower reflex to break symmetry, expand trust region.
- Mid Phase (steps 36-100): Airfoil has camber; gently fine-tune thickness distribution and suction peak position.
- Late Transition (steps 101-150): Micro-tune and slightly contract trust region for smooth local Bayesian convergence.
- After step 150: The optimization will transition entirely to pure Bayesian local exploitation.

Output ONLY a valid JSON object matching this schema:
{
  "thought": "1-2 sentence engineering rationale (under 40 words).",
  "length_scale_factor": float in [0.8, 1.8],
  "prob_perturb": float in [0.25, 0.75],
  "exploration_weight": float in [0.8, 1.8],
  "directional_strategy": {
    "upper_camber_shift": float in [-0.05, 0.15],
    "lower_camber_shift": float in [-0.05, 0.15],
    "thickness_shift": float in [-0.03, 0.04],
    "le_radius_shift": float in [-0.03, 0.03]
  }
}
"""


class OptimizedLLMMetaAgent:
    """针对 Qwen3.8-27B 宏观气动方向决策优化的元控制器智能体"""

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

        if HAS_OPENAI:
            self.client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=0,
            )
        else:
            self.client = None

    def build_user_prompt(self, state: Dict[str, Any]) -> str:
        """根据当前优化阶段动态构建高信息密度 prompt"""
        step = state["step"]
        best_cl = state.get("best_cl", 0.39)
        best_ld = state.get("best_ld", 65.8)

        if step <= 35 and best_cl < 0.48:
            phase_str = (
                "EARLY EXPLORATION (Aggressive Camber Needed): Baseline NACA 0012 is symmetric. "
                "You MUST apply bold positive upper camber shift (+0.07 ~ +0.12) and lower reflex (+0.05 ~ +0.10) "
                "to break symmetry and boost lift."
            )
        elif step <= 100:
            phase_str = (
                f"MID REFINEMENT: Airfoil already developed positive camber (CL={best_cl:.3f}, L/D={best_ld:.1f}). "
                "Apply gentle micro-adjustments (shifts in [-0.03, +0.03]) to optimize suction peak and reduce drag without over-thickening."
            )
        else:
            phase_str = (
                f"PRE-CONVERGENCE ALIGNMENT: L/D={best_ld:.1f}. Apply very small micro-tuning (shifts in [-0.02, +0.02]) "
                "and slightly contract trust region (length_scale_factor in [0.8, 1.0]) for smooth Bayesian convergence."
            )

        prompt = f"""=== Current Optimization State ===
- Phase: {phase_str}
- Airfoil: NACA0012 base | Re: {state['reynolds']:.2e} | Alpha: {state['alpha']} deg
- Progress: Step {step} / {state['max_evals']} (Remaining: {state['budget_left']})
- Initial L/D: {state['init_ld']:.2f} | Current Best L/D: {state['best_ld']:.2f}
- Aerodynamics: CL = {best_cl:.4f}, CD = {state.get('best_cd', 0.0):.5f}, t_max = {state.get('best_t_max', 0.12)*100:.2f}% (Target: 11%~15%)
- TuRBO State: TR Length L = {state['tr_length']:.4f}, Stagnation = {state['stagnation_count']}, Successes = {state['success_counter']}/{state['success_tolerance']}

Concisely decide meta-parameters and directional strategy. Output final JSON:"""
        return prompt

    def decide_action(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """向 LLM 请求下一阶段超参策略及宏观几何形变方向"""
        user_prompt = self.build_user_prompt(state)

        step = state.get("step", 16)
        is_early = (step <= 35 and state.get("best_cl", 0.39) < 0.48)
        default_action = {
            "thought": "Fallback to default TuRBO dynamics.",
            "length_scale_factor": 1.4 if is_early else 1.0,
            "prob_perturb": 0.6 if is_early else 0.5,
            "exploration_weight": 1.4 if is_early else 1.0,
            "directional_strategy": {
                "upper_camber_shift": 0.09 if is_early else 0.01,
                "lower_camber_shift": 0.07 if is_early else 0.01,
                "thickness_shift": 0.01,
                "le_radius_shift": 0.01,
            },
        }

        try:
            if self.client is not None:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": LLM_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7,
                    top_p=0.90,
                    max_tokens=2048,
                )
                raw_text = response.choices[0].message.content or ""
                clean = re.sub(r"```json\s*", "", raw_text)
                clean = re.sub(r"```\s*", "", clean).strip()
                match = re.search(r"\{.*\}", clean, re.DOTALL)
                if match:
                    data = json.loads(match.group(0))
                    scale_L = float(np.clip(data.get("length_scale_factor", 1.0), 0.8, 1.8))
                    prob_p = float(np.clip(data.get("prob_perturb", 0.5), 0.2, 0.8))
                    exp_w = float(np.clip(data.get("exploration_weight", 1.0), 0.8, 1.8))
                    strat = data.get("directional_strategy", {})
                    return {
                        "thought": str(data.get("thought", "")),
                        "length_scale_factor": scale_L,
                        "prob_perturb": prob_p,
                        "exploration_weight": exp_w,
                        "directional_strategy": {
                            "upper_camber_shift": float(np.clip(strat.get("upper_camber_shift", 0.0), -0.05, 0.15)),
                            "lower_camber_shift": float(np.clip(strat.get("lower_camber_shift", 0.0), -0.05, 0.15)),
                            "thickness_shift": float(np.clip(strat.get("thickness_shift", 0.0), -0.04, 0.05)),
                            "le_radius_shift": float(np.clip(strat.get("le_radius_shift", 0.0), -0.03, 0.03)),
                        },
                    }
            return default_action
        except Exception as e:
            print(f"        [LLM Warning] Request encountered '{type(e).__name__}: {e}'. Falling back to safe default.")
            return default_action
