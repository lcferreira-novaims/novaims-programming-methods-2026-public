import builtins

from pm_labs import chatbot


def test_chatbot_sends_messages_and_stops_on_exit(monkeypatch, capsys):
    user_inputs = iter(["hello", "exit"])
    monkeypatch.setattr(builtins, "input", lambda _: next(user_inputs))
    calls = []

    def fake_chat(*, model, messages):
        calls.append((model, [message.copy() for message in messages]))
        return {"message": {"content": "hello back"}}

    monkeypatch.setattr(chatbot.ollama, "chat", fake_chat)

    chatbot.main()

    assert calls == [
        (
            "llama3.2:1b",
            [{"role": "user", "content": "hello"}],
        )
    ]
    assert "AI: hello back" in capsys.readouterr().out


def test_chatbot_accepts_quit_without_calling_model(monkeypatch):
    monkeypatch.setattr(builtins, "input", lambda _: "quit")

    def unexpected_chat(**kwargs):
        raise AssertionError("Ollama should not be called when quitting")

    monkeypatch.setattr(chatbot.ollama, "chat", unexpected_chat)

    chatbot.main()
