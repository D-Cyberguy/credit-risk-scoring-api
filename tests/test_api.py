from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200


def test_predict():
    payload = {
        "person_age": 30,
        "person_income": 60000,
        "person_home_ownership": "RENT",
        "person_emp_length": 4,
        "loan_intent": "PERSONAL",
        "loan_grade": "B",
        "loan_amnt": 12000,
        "loan_int_rate": 12.0,
        "loan_percent_income": 0.2,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 7
    }

    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert "decision" in r.json()
