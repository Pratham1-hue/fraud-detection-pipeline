import random
from locust import HttpUser, task, between

class FraudPipelineLoadTest(HttpUser):
    wait_time = between(0.01, 0.05)

    @task
    def predict_transaction(self):
        payload = {
            "transaction_id": f"tx_{random.randint(100000, 999999)}",
            "user_id": f"usr_{random.randint(1000, 9999)}",
            "amount": round(random.uniform(5.0, 2500.0), 2),
            "time_delta": round(random.uniform(0, 86400), 1),
            "features": [random.gauss(0,1) for _ in range(28)]
        }


        self.client.post("/predict", json=payload)
        