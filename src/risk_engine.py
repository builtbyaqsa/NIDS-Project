class RiskEngine:
    """
    Calculates risk levels and assigns recommended security actions 
    based on NIDS model predictions and anomaly probabilities.
    """
    def __init__(self):
        pass

    def evaluate_risk(self, prediction: int, confidence: float = 1.0) -> dict:
        """
        Evaluates risk score based on whether traffic is normal (0) or attack (1).
        """
        if prediction == 0:
            risk_level = "LOW"
            action = "Allow traffic / Monitor normally"
            score = 10.0 * confidence
        else:
            if confidence > 0.85:
                risk_level = "CRITICAL / HIGH"
                action = "Block IP / Alert Security Operations Center (SOC)"
                score = 90.0 * confidence
            else:
                risk_level = "MEDIUM"
                action = "Flag for deep inspection / Rate limit"
                score = 50.0 * confidence

        return {
            "risk_level": risk_level,
            "recommended_action": action,
            "risk_score": round(score, 2)
        }

if __name__ == "__main__":
    # Quick test of the Risk Engine
    engine = RiskEngine()
    print("Test Normal Traffic:", engine.evaluate_risk(0))
    print("Test Attack Traffic:", engine.evaluate_risk(1, confidence=0.92))