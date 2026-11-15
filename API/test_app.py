from fastapi.testclient import TestClient
from main import app
clinet= TestClient(app)
def test_read_root():
    response= clinet.get("/")
    assert response.status_code in [200,404]
def test_health_check():
    response= clinet.get("/health")
    assert response.status_code in [200,503]