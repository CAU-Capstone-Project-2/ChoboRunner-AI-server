# -*- coding: utf-8 -*-
"""healthcheck 엔드포인트 통합 테스트 (docs/docker-container §6).

``GET /healthz`` — liveness probe. 분석 의존성(모델 로드·연결 상태)이 없는
즉답 엔드포인트이므로 model 파일(.task) 없이도 동작해야 한다. 따라서
test_server_ws.py와 달리 model 부재 skip 마크를 두지 않는다.
"""
from __future__ import annotations

from fastapi.testclient import TestClient


def _make_app():
    from server.main import create_app

    return create_app()


def test_healthz_returns_ok():
    """GET /healthz → 200 + {"status": "ok"}."""
    with TestClient(_make_app()) as client:
        resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_healthz_independent_of_model():
    """healthcheck는 분석 의존성이 없다 — 반복 호출에도 동일한 즉답.

    Docker HEALTHCHECK가 30초 주기로 반복 호출하는 사용 패턴을 모사한다.
    """
    with TestClient(_make_app()) as client:
        for _ in range(3):
            resp = client.get("/healthz")
            assert resp.status_code == 200
            assert resp.json()["status"] == "ok"


def test_healthz_no_medical_disclaimer_flag():
    """healthcheck 응답은 분석 결과가 아니므로 면책 플래그 대상이 아니다.

    docs/docker-container §6-1 — reference_feedback_only는 분석 JSON 응답에만
    붙는다. 운영 응답에 섞이지 않음을 회귀로 고정한다.
    """
    with TestClient(_make_app()) as client:
        body = client.get("/healthz").json()
    assert "reference_feedback_only" not in body
