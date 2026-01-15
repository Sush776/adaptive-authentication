# src/metrics.py

from threading import Lock

class MetricsStore:
    def __init__(self):
        self.requests_total = 0
        self.mfa_triggered_total = 0
        self.prediction_errors_total = 0
        self.total_risk_score = 0.0
        self._lock = Lock()

    def record_prediction(self, risk_score: float, mfa_required: bool):
        with self._lock:
            self.requests_total += 1
            self.total_risk_score += risk_score
            if mfa_required:
                self.mfa_triggered_total += 1

    def record_error(self):
        with self._lock:
            self.prediction_errors_total += 1

    def get_metrics(self):
        with self._lock:
            avg_risk = (
                self.total_risk_score / self.requests_total
                if self.requests_total > 0 else 0.0
            )
            return {
                "requests_total": self.requests_total,
                "mfa_triggered_total": self.mfa_triggered_total,
                "prediction_errors_total": self.prediction_errors_total,
                "avg_risk_score": round(avg_risk, 4)
            }


# global instance used by API
metrics_store = MetricsStore()
