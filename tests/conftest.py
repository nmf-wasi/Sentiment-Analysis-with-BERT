import time
import pytest
from fastapi.testclient import TestClient
from src.main import app


@pytest.fixture(
    scope="module"
)  # the fixture runs once for the whole file, so the model loads once instead of once per test.
def client():
    with TestClient(app) as c:
        deadline = time.time() + 120
        while time.time() < deadline:
            if c.get("/health").status_code == 200:
                break
            time.sleep(1)
        else:
            pytest.fail("Model never became ready!")

        yield c
