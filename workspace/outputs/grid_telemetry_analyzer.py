"""
GridFlow AI Verified Production Artifact: Smart Grid Telemetry & Anomaly Detector
Prompt: Build a Smart Grid Substation Telemetry Anomaly Detector in Python
Verified Citations: DOC-CODE-002
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class SubstationReading:
    substation_id: str
    timestamp_ms: int
    load_mw: float
    voltage_kv: float
    frequency_hz: float


class GridTelemetryAnalyzer:
    """Real-time z-score anomaly detector and load-balancing advisor for grid substations."""

    def __init__(self, z_threshold: float = 2.2, nominal_freq_hz: float = 50.0) -> None:
        self.z_threshold = z_threshold
        self.nominal_freq_hz = nominal_freq_hz

    def analyze_batch(self, readings: List[SubstationReading]) -> Dict[str, Any]:
        if not readings:
            return {"status": "EMPTY", "anomalies": [], "summary": {}}

        loads = [r.load_mw for r in readings]
        mean_load = sum(loads) / len(loads)
        variance = sum((x - mean_load) ** 2 for x in loads) / max(1, len(loads))
        std_load = math.sqrt(variance) or 1e-6

        anomalies: List[Dict[str, Any]] = []
        for r in readings:
            z_score = abs(r.load_mw - mean_load) / std_load
            freq_drift = abs(r.frequency_hz - self.nominal_freq_hz)
            if z_score >= self.z_threshold or freq_drift >= 0.4:
                anomalies.append({
                    "substation_id": r.substation_id,
                    "timestamp_ms": r.timestamp_ms,
                    "load_mw": round(r.load_mw, 2),
                    "z_score": round(z_score, 3),
                    "frequency_drift_hz": round(freq_drift, 3),
                    "recommended_action": "SHED_NON_CRITICAL_LOAD" if r.load_mw > mean_load else "REROUTE_FEEDER",
                })

        return {
            "status": "ALERT" if anomalies else "NOMINAL",
            "mean_load_mw": round(mean_load, 2),
            "std_load_mw": round(std_load, 2),
            "total_readings": len(readings),
            "anomaly_count": len(anomalies),
            "anomalies": anomalies,
        }


if __name__ == "__main__":
    sample = [
        SubstationReading("SUB-NORTH-01", 1700000100, 142.5, 132.0, 50.01),
        SubstationReading("SUB-NORTH-02", 1700000200, 145.0, 131.8, 49.98),
        SubstationReading("SUB-EAST-03", 1700000300, 139.2, 132.1, 50.00),
        SubstationReading("SUB-WEST-04", 1700000400, 141.8, 132.0, 50.02),
        SubstationReading("SUB-SOUTH-99", 1700000500, 310.4, 126.2, 49.42),
    ]
    analyzer = GridTelemetryAnalyzer(z_threshold=1.8)
    print(analyzer.analyze_batch(sample))
