"""HTTP contract checks that need no model access."""
from app import main


def test_health_reports_connected_vector_db(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "healthy"
    assert body["services"]["vector_db"] == "connected"


def test_health_reports_disconnected_vector_db(client):
    main.app.state.qdrant.up = False
    body = client.get("/api/v1/health").json()
    assert body["services"]["vector_db"] == "disconnected"


def test_empty_request_rejected(client):
    response = client.post("/api/v1/claims/triage")
    assert response.status_code == 400
    assert "photo" in response.json()["message"]


def test_blank_description_rejected(client):
    response = client.post("/api/v1/claims/triage", files={"description": (None, "   ")})
    assert response.status_code == 400


def test_non_image_upload_rejected(client):
    response = client.post("/api/v1/claims/triage",
                           files={"image": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400
    assert "Unsupported image type" in response.json()["message"]


def test_unknown_policy_rejected(client):
    response = client.post("/api/v1/claims/triage",
                           files={"description": (None, "Pipe burst in kitchen"),
                                  "policy_type": (None, "gold_plated")})
    assert response.status_code == 400


def test_unknown_claim_returns_404(client):
    assert client.get("/api/v1/claims/does-not-exist").status_code == 404
    assert client.get("/api/v1/agent/trace/does-not-exist").status_code == 404


def test_saved_report_and_trace_round_trip(client):
    store = main.app.state.store
    store.save("CLM-TEST", {"claim_id": "CLM-TEST"}, {"claim_id": "CLM-TEST", "steps": []})
    assert client.get("/api/v1/claims/CLM-TEST").json()["claim_id"] == "CLM-TEST"
    assert client.get("/api/v1/agent/trace/CLM-TEST").json()["claim_id"] == "CLM-TEST"
