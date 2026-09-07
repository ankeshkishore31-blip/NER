"""
Physics-Informed Slope Stability & Machine Learning Engine for NER Landslides.
Implements the Infinite Slope Model for Factor of Safety (FS) coupled with
statistical susceptibility estimation.
"""

import math
import numpy as np
from typing import Dict, Any

class SlopeStabilityPINN:
    def __init__(self, gamma_soil: float = 18.5, gamma_water: float = 9.81):
        """
        gamma_soil: Bulk unit weight of soil in kN/m3
        gamma_water: Unit weight of water (9.81 kN/m3)
        """
        self.gamma_soil = gamma_soil
        self.gamma_water = gamma_water

    def calculate_factor_of_safety(
        self,
        slope_deg: float,
        cohesion_kpa: float,
        friction_deg: float,
        soil_depth_m: float,
        water_table_ratio: float
    ) -> float:
        """
        Differentiable Infinite Slope Stability Equation:
        FS = [c' + (gamma * z * cos^2(theta) - gamma_w * h_w * cos^2(theta)) * tan(phi')] /
             [gamma * z * sin(theta) * cos(theta)]
        """
        # Constrain inputs to physically meaningful ranges
        theta_rad = math.radians(max(slope_deg, 1.0))
        phi_rad = math.radians(max(friction_deg, 5.0))
        z = max(soil_depth_m, 0.5)
        m = min(max(water_table_ratio, 0.0), 1.0) # h_w / z
        h_w = m * z

        cos_theta = math.cos(theta_rad)
        sin_theta = math.sin(theta_rad)

        total_normal_stress = self.gamma_soil * z * (cos_theta ** 2)
        pore_water_pressure = self.gamma_water * h_w * (cos_theta ** 2)
        effective_normal_stress = max(total_normal_stress - pore_water_pressure, 0.01)

        resisting_shear_strength = cohesion_kpa + effective_normal_stress * math.tan(phi_rad)
        driving_shear_stress = self.gamma_soil * z * sin_theta * cos_theta

        if driving_shear_stress <= 0.001:
            return 9.99

        fs = resisting_shear_strength / driving_shear_stress
        return round(float(fs), 3)

    def predict_hybrid_risk(
        self,
        slope_deg: float,
        cohesion_kpa: float,
        friction_deg: float,
        soil_depth_m: float,
        water_table_ratio: float,
        elevation_m: float = 1450.0,
        rainfall_cumulative_3d_mm: float = 120.0,
        distance_to_road_m: float = 45.0,
        pore_pressure_kpa: float = 25.0
    ) -> Dict[str, Any]:
        """
        Hybrid Ensemble: Couples Physics Factor of Safety with
        environmental empirical conditioning factors.
        """
        fs = self.calculate_factor_of_safety(
            slope_deg=slope_deg,
            cohesion_kpa=cohesion_kpa,
            friction_deg=friction_deg,
            soil_depth_m=soil_depth_m,
            water_table_ratio=water_table_ratio
        )

        # Physics-based failure probability mapping (FS < 1.0 => high failure probability)
        # Using a generalized sigmoid centered at FS = 1.0
        # P_phys = 1 / (1 + exp(4 * (FS - 1.0)))
        p_physics = 1.0 / (1.0 + math.exp(min(max(4.2 * (fs - 1.0), -10), 10)))

        # Statistical conditioning factor score
        # Rainfall saturation weight
        rain_score = min(rainfall_cumulative_3d_mm / 250.0, 1.0)
        # Slope steepness heuristic
        slope_score = min(slope_deg / 60.0, 1.0)
        # Road proximity cut-slope effect
        road_cut_score = max(1.0 - (distance_to_road_m / 200.0), 0.1)
        # Pore pressure factor
        pore_score = min(pore_pressure_kpa / 60.0, 1.0)

        p_statistical = (
            0.40 * rain_score +
            0.30 * slope_score +
            0.15 * road_cut_score +
            0.15 * pore_score
        )

        # Hybrid ensemble probability (60% physics, 40% statistical)
        probability_of_failure = round(0.60 * p_physics + 0.40 * p_statistical, 4)

        # Hazard Level Classification
        if probability_of_failure >= 0.80 or fs < 1.0:
            hazard_level = "RED_CRITICAL"
            color = "#EF4444"
            action = "IMMEDIATE EVACUATION MANDATED: High probability of rapid mass movement."
        elif probability_of_failure >= 0.60:
            hazard_level = "ORANGE_HIGH"
            color = "#F97316"
            action = "HIGH VIGILANCE: Prepare evacuation shelters, dispatch BRO road inspection teams."
        elif probability_of_failure >= 0.35:
            hazard_level = "YELLOW_MODERATE"
            color = "#EAB308"
            action = "WATCH & MONITOR: Slope creep detected; monitor telemetry feeds."
        else:
            hazard_level = "GREEN_LOW"
            color = "#22C55E"
            action = "NORMAL: Slope structurally stable within safe hydrological envelope."

        return {
            "factor_of_safety": fs,
            "probability_of_failure": probability_of_failure,
            "hazard_level": hazard_level,
            "badge_color": color,
            "action_recommendation": action,
            "diagnostics": {
                "p_physics": round(p_physics, 4),
                "p_statistical": round(p_statistical, 4),
                "rain_saturation_pct": round(rain_score * 100, 1),
                "pore_pressure_kpa": pore_pressure_kpa
            }
        }
