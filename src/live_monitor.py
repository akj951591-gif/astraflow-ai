from dataclasses import dataclass
from datetime import datetime


@dataclass
class LiveTrafficInput:
    current_speed_kmph: float
    normal_speed_kmph: float
    vehicle_density: int
    queue_length_m: float
    rainfall_mm: float
    crowd_level: int
    road_blocked_percent: float
    emergency_vehicle_required: bool


class LiveTrafficMonitor:
    def calculate_live_risk(self, data: LiveTrafficInput):
        normal_speed = max(float(data.normal_speed_kmph), 1)

        speed_reduction = max(
            0,
            min(
                100,
                ((normal_speed - data.current_speed_kmph) / normal_speed) * 100
            )
        )

        speed_score = speed_reduction * 0.28
        density_score = min(data.vehicle_density / 150 * 100, 100) * 0.18
        queue_score = min(data.queue_length_m / 1000 * 100, 100) * 0.18
        rainfall_score = min(data.rainfall_mm / 50 * 100, 100) * 0.08
        crowd_score = min(data.crowd_level, 100) * 0.12
        blockage_score = min(data.road_blocked_percent, 100) * 0.16

        risk_score = (
            speed_score
            + density_score
            + queue_score
            + rainfall_score
            + crowd_score
            + blockage_score
        )

        if data.emergency_vehicle_required:
            risk_score += 8

        risk_score = round(min(max(risk_score, 0), 100), 1)

        if risk_score >= 80:
            risk_level = "CRITICAL"
            alert_code = "RED"
            update_interval = 1

        elif risk_score >= 60:
            risk_level = "HIGH"
            alert_code = "ORANGE"
            update_interval = 2

        elif risk_score >= 35:
            risk_level = "MODERATE"
            alert_code = "YELLOW"
            update_interval = 5

        else:
            risk_level = "LOW"
            alert_code = "GREEN"
            update_interval = 10

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "alert_code": alert_code,
            "speed_reduction_percent": round(speed_reduction, 1),
            "estimated_delay_min": self._estimate_delay(
                risk_score,
                data.queue_length_m,
                speed_reduction
            ),
            "recommended_action": self._recommended_action(
                risk_score,
                data
            ),
            "alerts": self._generate_alerts(
                data,
                speed_reduction,
                risk_score
            ),
            "update_interval_min": update_interval,
            "generated_at": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        }

    @staticmethod
    def _estimate_delay(risk_score, queue_length_m, speed_reduction):
        delay = (
            risk_score * 0.55
            + queue_length_m / 40
            + speed_reduction * 0.25
        )

        return round(min(max(delay, 0), 240), 1)

    @staticmethod
    def _recommended_action(risk_score, data):
        actions = []

        if risk_score >= 80:
            actions.extend([
                "Activate emergency diversion plan",
                "Deploy additional traffic police immediately",
                "Enable manual traffic signal override",
                "Issue public traffic advisory",
                "Notify emergency response unit"
            ])

        elif risk_score >= 60:
            actions.extend([
                "Activate partial diversion",
                "Deploy officers at major intersections",
                "Place temporary barricades",
                "Increase CCTV monitoring frequency"
            ])

        elif risk_score >= 35:
            actions.extend([
                "Send field team for verification",
                "Prepare diversion route",
                "Monitor queue growth",
                "Adjust nearby signal timing"
            ])

        else:
            actions.extend([
                "Continue routine monitoring",
                "No immediate intervention required"
            ])

        if data.road_blocked_percent >= 60:
            actions.append("Consider complete temporary road closure")

        if data.emergency_vehicle_required:
            actions.append("Create emergency green corridor")

        if data.rainfall_mm >= 25:
            actions.append("Deploy waterlogging response team")

        return actions

    @staticmethod
    def _generate_alerts(data, speed_reduction, risk_score):
        alerts = []

        if speed_reduction >= 70:
            alerts.append({
                "severity": "CRITICAL",
                "message": "Traffic speed has reduced by more than 70%."
            })

        if data.queue_length_m >= 700:
            alerts.append({
                "severity": "HIGH",
                "message": "Vehicle queue has exceeded 700 metres."
            })

        if data.vehicle_density >= 120:
            alerts.append({
                "severity": "HIGH",
                "message": "Vehicle density is approaching road capacity."
            })

        if data.crowd_level >= 75:
            alerts.append({
                "severity": "HIGH",
                "message": "Large crowd spillover detected near the road."
            })

        if data.road_blocked_percent >= 70:
            alerts.append({
                "severity": "CRITICAL",
                "message": "Most of the carriageway is currently blocked."
            })

        if data.rainfall_mm >= 30:
            alerts.append({
                "severity": "MODERATE",
                "message": "Heavy rainfall may further reduce road capacity."
            })

        if data.emergency_vehicle_required:
            alerts.append({
                "severity": "CRITICAL",
                "message": "Emergency vehicle access is required."
            })

        if not alerts and risk_score < 35:
            alerts.append({
                "severity": "NORMAL",
                "message": "Traffic conditions are within normal limits."
            })

        return alerts