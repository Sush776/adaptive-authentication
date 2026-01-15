import csv
import os

LOG_FILE = "data/prediction_logs.csv"

def log_prediction(event, risk_prob, mfa_decision):
    file_exists = os.path.isfile(LOG_FILE)

    # IMPORTANT: add encoding="utf-8"
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "keystroke", "mouse_speed", "failed_attempts_last_24h",
                "new_device", "new_location", "risky_ip",
                "device", "browser", "ip_country", "timestamp",
                "risk_prob", "mfa_decision"
            ])

        writer.writerow([
            event.keystroke,
            event.mouse_speed,
            event.failed_attempts_last_24h,
            event.new_device,
            event.new_location,
            event.risky_ip,
            event.device,
            event.browser,
            event.ip_country,
            event.timestamp,
            risk_prob,
            mfa_decision
        ])
