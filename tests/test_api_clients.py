from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from pm_labs import api_clients
from pm_labs.api_clients import (
    APIClient,
    ChatSession,
    OllamaClient,
    OllamaResponse,
    WorldBankClient,
)


def test_api_client_returns_json(monkeypatch):
    response = SimpleNamespace(status_code=200, text="", json=lambda: {"ok": True})
    request = Mock(return_value=response)
    monkeypatch.setattr(api_clients.requests, "request", request)
    client = APIClient("https://example.test")

    result = client._request("GET", "/items", params={"page": 1})

    assert result == {"ok": True}
    request.assert_called_once_with(
        "GET", "https://example.test/items", params={"page": 1}
    )


def test_api_client_raises_for_non_200_response(monkeypatch):
    response = SimpleNamespace(status_code=503, text="unavailable")
    monkeypatch.setattr(
        api_clients.requests, "request", Mock(return_value=response)
    )
    client = APIClient("https://example.test")

    with pytest.raises(RuntimeError, match="status 503: unavailable"):
        client._request("GET", "/items")


def test_world_bank_client_parses_population(monkeypatch):
    client = WorldBankClient()
    request = Mock(
        return_value=[
            {},
            [{"country": {"value": "Portugal"}, "date": "2024", "value": 10_000_000}],
        ]
    )
    monkeypatch.setattr(client, "_request", request)

    result = client.get_population("PRT")

    assert result.country == "Portugal"
    assert result.date == "2024"
    assert result.value == 10_000_000
    request.assert_called_once_with(
        "GET",
        "/country/PRT/indicator/SP.POP.TOTL",
        params={"format": "json"},
    )


def test_ollama_client_formats_prompt_and_sends_generation_request(monkeypatch):
    client = OllamaClient(model="test-model")
    request = Mock(return_value={"response": "A response"})
    monkeypatch.setattr(client, "_request", request)

    prompt = client.format_prompt("Portugal", 10_000_000)
    result = client.generate(prompt)

    assert "10,000,000" in prompt
    assert "Portugal" in prompt
    assert result == {"response": "A response"}
    request.assert_called_once_with(
        "POST",
        "/api/generate",
        json={"model": "test-model", "prompt": prompt, "stream": False},
    )


def test_ollama_client_validates_generation_response(monkeypatch):
    client = OllamaClient()
    monkeypatch.setattr(
        client,
        "generate",
        Mock(
            return_value={
                "model": "test-model",
                "response": "hello",
                "done": True,
            }
        ),
    )

    result = client.generate_validated("prompt")

    assert isinstance(result, OllamaResponse)
    assert result.response == "hello"


def test_ollama_client_reports_invalid_generation_response(monkeypatch):
    client = OllamaClient()
    monkeypatch.setattr(client, "generate", Mock(return_value={"model": "test"}))

    with pytest.raises(ValueError, match="response"):
        client.generate_validated("prompt")


def test_chat_session_keeps_turn_history():
    client = Mock()
    client.chat.side_effect = ["first reply", "second reply"]
    session = ChatSession(client)

    assert session.send("first question") == "first reply"
    assert session.send("second question") == "second reply"
    assert session.messages == [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first reply"},
        {"role": "user", "content": "second question"},
        {"role": "assistant", "content": "second reply"},
    ]


def test_ollama_client_chat_uses_message_history(monkeypatch):
    client = OllamaClient(model="test-model")
    chat = Mock(return_value={"message": {"content": "hello"}})
    monkeypatch.setattr(api_clients.ollama, "chat", chat)
    messages = [{"role": "user", "content": "hello"}]

    assert client.chat(messages) == "hello"
    chat.assert_called_once_with(model="test-model", messages=messages)
