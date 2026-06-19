from dataclasses import dataclass
from typing import Dict, Any

import pandas as pd


@dataclass
class ScenarioPlan:
    name: str
    officers: int
    barricades: int
    diversion: bool
    road_closure: bool
    signal_override: bool
    public_advisory: bool


class ScenarioSimulator:
    """
    Simulates how different traffic-control plans may reduce
    disruption duration and congestion risk.

    This is a decision-support simulation, not a replacement
    for the trained duration prediction model.
    """

    @staticmethod
    def _clamp(value, minimum, maximum):
        return max(minimum, min(value, maximum))

    def simulate(
        self,
        baseline_duration_min: float,
        baseline_risk_score: float,
        officers: int,
        barricades: int,
        diversion: bool,
        road_closure: bool,
        signal_override: bool,
        public_advisory: bool,
    ) -> Dict[str, Any]:

        baseline_duration_min = max(
            float(baseline_duration_min),
            1.0
        )

        baseline_risk_score = self._clamp(
            float(baseline_risk_score),
            0,
            100
        )

        officers = max(int(officers), 0)
        barricades = max(int(barricades), 0)

        required_officers = max(
            6,
            round(baseline_risk_score * 0.35)
        )

        required_barricades = max(
            2,
            round(baseline_risk_score * 0.18)
        )

        officer_ratio = min(
            officers / required_officers,
            1.5
        )

        barricade_ratio = min(
            barricades / required_barricades,
            1.5
        )

        officer_effect = min(
            officer_ratio * 0.14,
            0.18
        )

        barricade_effect = min(
            barricade_ratio * 0.09,
            0.12
        )

        diversion_effect = 0.16 if diversion else 0.0
        closure_effect = 0.12 if road_closure else 0.0
        signal_effect = 0.10 if signal_override else 0.0
        advisory_effect = 0.04 if public_advisory else 0.0

        individual_effects = [
            officer_effect,
            barricade_effect,
            diversion_effect,
            closure_effect,
            signal_effect,
            advisory_effect,
        ]

        remaining_fraction = 1.0

        for effect in individual_effects:
            remaining_fraction *= 1.0 - effect

        total_reduction = 1.0 - remaining_fraction
        total_reduction = min(total_reduction, 0.60)

        officer_shortage = max(
            required_officers - officers,
            0
        )

        barricade_shortage = max(
            required_barricades - barricades,
            0
        )

        officer_shortage_ratio = (
            officer_shortage / required_officers
            if required_officers > 0
            else 0
        )

        barricade_shortage_ratio = (
            barricade_shortage / required_barricades
            if required_barricades > 0
            else 0
        )

        shortage_penalty = (
            baseline_duration_min
            * (
                officer_shortage_ratio * 0.15
                + barricade_shortage_ratio * 0.08
            )
        )

        estimated_duration = (
            baseline_duration_min
            * (1.0 - total_reduction)
            + shortage_penalty
        )

        estimated_duration = max(
            estimated_duration,
            5
        )

        estimated_risk = (
            baseline_risk_score
            * (1.0 - total_reduction * 0.80)
        )

        estimated_risk += (
            officer_shortage_ratio * 10
            + barricade_shortage_ratio * 6
        )

        estimated_risk = self._clamp(
            estimated_risk,
            0,
            100
        )

        duration_reduction = max(
            baseline_duration_min - estimated_duration,
            0
        )

        duration_reduction_percent = (
            duration_reduction
            / baseline_duration_min
            * 100
        )

        resource_cost_index = (
            officers * 1.0
            + barricades * 0.55
            + (12 if diversion else 0)
            + (15 if road_closure else 0)
            + (8 if signal_override else 0)
            + (3 if public_advisory else 0)
        )

        effectiveness_score = (
            duration_reduction_percent * 0.55
            + (
                baseline_risk_score
                - estimated_risk
            ) * 0.45
        )

        efficiency_score = (
            effectiveness_score
            / max(resource_cost_index, 1)
            * 20
        )

        efficiency_score = self._clamp(
            efficiency_score,
            0,
            100
        )

        if estimated_risk >= 80:
            risk_level = "CRITICAL"
        elif estimated_risk >= 60:
            risk_level = "HIGH"
        elif estimated_risk >= 35:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        recommendations = []

        if officers < required_officers:
            recommendations.append(
                f"Add at least "
                f"{required_officers - officers} more officers."
            )

        if barricades < required_barricades:
            recommendations.append(
                f"Add at least "
                f"{required_barricades - barricades} more barricades."
            )

        if baseline_risk_score >= 60 and not diversion:
            recommendations.append(
                "Activate a diversion route."
            )

        if baseline_risk_score >= 75 and not signal_override:
            recommendations.append(
                "Enable manual traffic-signal control."
            )

        if baseline_risk_score >= 70 and not public_advisory:
            recommendations.append(
                "Issue a public traffic advisory."
            )

        if estimated_risk < 35:
            recommendations.append(
                "Selected response plan should maintain manageable traffic."
            )

        if not recommendations:
            recommendations.append(
                "Selected plan provides sufficient operational coverage."
            )

        return {
            "baseline_duration_min": round(
                baseline_duration_min,
                1
            ),
            "estimated_duration_min": round(
                estimated_duration,
                1
            ),
            "duration_saved_min": round(
                duration_reduction,
                1
            ),
            "duration_reduction_percent": round(
                duration_reduction_percent,
                1
            ),
            "baseline_risk_score": round(
                baseline_risk_score,
                1
            ),
            "estimated_risk_score": round(
                estimated_risk,
                1
            ),
            "risk_level": risk_level,
            "required_officers": required_officers,
            "required_barricades": required_barricades,
            "resource_cost_index": round(
                resource_cost_index,
                1
            ),
            "efficiency_score": round(
                efficiency_score,
                1
            ),
            "recommendations": recommendations,
        }

    def compare_plans(
        self,
        baseline_duration_min: float,
        baseline_risk_score: float,
        plans: list[ScenarioPlan],
    ) -> pd.DataFrame:

        rows = []

        for plan in plans:
            result = self.simulate(
                baseline_duration_min=
                    baseline_duration_min,

                baseline_risk_score=
                    baseline_risk_score,

                officers=plan.officers,
                barricades=plan.barricades,
                diversion=plan.diversion,
                road_closure=plan.road_closure,
                signal_override=plan.signal_override,
                public_advisory=plan.public_advisory,
            )

            rows.append({
                "Plan": plan.name,
                "Officers": plan.officers,
                "Barricades": plan.barricades,
                "Diversion": (
                    "Yes"
                    if plan.diversion
                    else "No"
                ),
                "Road Closure": (
                    "Yes"
                    if plan.road_closure
                    else "No"
                ),
                "Estimated Duration": result[
                    "estimated_duration_min"
                ],
                "Duration Saved": result[
                    "duration_saved_min"
                ],
                "Risk After Plan": result[
                    "estimated_risk_score"
                ],
                "Risk Level": result[
                    "risk_level"
                ],
                "Cost Index": result[
                    "resource_cost_index"
                ],
                "Efficiency Score": result[
                    "efficiency_score"
                ],
            })

        return pd.DataFrame(rows)