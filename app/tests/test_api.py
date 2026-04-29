#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ..main import app


@pytest.fixture
def client():
    # use "with" statement to run "startup" event of FastAPI
    with TestClient(app) as c:
        yield c


def test_main_predict(client):
    """
    Test predction response
    """

    root_dir = Path(__file__).resolve().parents[2]
    image_path = root_dir / "Test_Set_Folder" / "soybeans" / "soybeans287.jpg"
    assert image_path.exists(), f"Missing test image: {image_path}"

    with image_path.open("rb") as f:
        files = {"file": (image_path.name, f, "image/jpeg")}
        response = client.post("/api/v1/predict", files=files)

    try:
        assert response.status_code == 200
        reponse_json = response.json()
        assert reponse_json['error'] is False
        assert isinstance(reponse_json['results']['label'], str)
        assert isinstance(reponse_json['results']['confidence'], float)
        assert isinstance(reponse_json['results']['top_labels'], list)
        assert isinstance(reponse_json['results']['top_confidences'], list)
        assert len(reponse_json['results']['top_labels']) == len(reponse_json['results']['top_confidences'])

    except AssertionError:
        print(response.status_code)
        print(response.json())
        raise
