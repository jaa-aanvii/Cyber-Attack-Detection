
SEVERITY_MAP = {
    "Benign": 0.0,
    "Recon": 0.4,
    "BruteForce": 0.7,
    "Spoofing": 0.75,
    "Web": 0.8,
    "DoS": 0.85,
    "DDoS": 0.9,
    "Mirai": 0.95,
}


def evaluate_alert(attack_type, confidence, threshold=0.7):
    severity_score = SEVERITY_MAP[attack_type] * confidence

    if severity_score >= threshold and attack_type != "Benign":
        message = "ALERT: Severity score exceeds threshold!"
    else:
        message = "No alert: Severity score is below threshold."

    print(f"Type of Attack: {attack_type}")
    print(f"Severity Score: {severity_score:.2f}")
    print(f"Message: {message}")


if __name__ == "__main__":
    # Example: DDoS prediction with 85% confidence
    evaluate_alert("DDoS", confidence=0.85, threshold=0.7)
    evaluate_alert("Benign", confidence=0.95, threshold=0.7)

