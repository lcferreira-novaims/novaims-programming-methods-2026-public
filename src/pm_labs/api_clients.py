"""API clients from Lab 3 (Object Oriented Programming Part I).

``APIClient`` centralizes request/status-code handling; ``WorldBankClient``
and ``OllamaClient`` are thin subclasses over the World Bank REST API and a
local Ollama server, respectively. ``ChatSession`` wraps ``OllamaClient`` in
a small stateful chat loop.
"""

from typing import Optional

import ollama
import requests
from pydantic import BaseModel, ValidationError


class WorldBankPopulation(BaseModel):
    country: str
    date: str
    value: Optional[int]


class OllamaResponse(BaseModel):
    model: str
    response: str
    done: bool
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None


class APIClient:
    """Base class for talking to a JSON HTTP API."""

    def __init__(self, base_url: str):
        self.base_url = base_url

    def _request(self, method: str, endpoint: str, **kwargs) -> dict:
        url = f"{self.base_url}{endpoint}"
        response = requests.request(method, url, **kwargs)

        if response.status_code != 200:
            raise RuntimeError(
                f"Request to {url} failed with status {response.status_code}: "
                f"{response.text}"
            )

        return response.json()

    def format_prompt(self, *args, **kwargs) -> str:
        raise NotImplementedError


class WorldBankClient(APIClient):
    def __init__(self):
        super().__init__(base_url="https://api.worldbank.org/v2")

    def get_population(self, country_code: str) -> WorldBankPopulation:
        endpoint = f"/country/{country_code}/indicator/SP.POP.TOTL"
        raw = self._request("GET", endpoint, params={"format": "json"})

        # raw is [metadata_dict, data_list] — most recent year is data_list[0]
        data_entry = raw[1][0]

        return WorldBankPopulation(
            country=data_entry["country"]["value"],
            date=data_entry["date"],
            value=data_entry["value"],
        )


class OllamaClient(APIClient):
    def __init__(self, model: str = "llama3.2:1b"):
        super().__init__(base_url="http://localhost:11434")
        self.model = model

    def format_prompt(self, country: str, population: int) -> str:
        return (
            f"The population of {country} is approximately {population:,}. "
            f"In one or two sentences, comment on what this figure tells us "
            f"about the country."
        )

    def generate(self, prompt: str) -> dict:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        return self._request("POST", "/api/generate", json=payload)

    def generate_validated(self, prompt: str) -> OllamaResponse:
        raw = self.generate(prompt)
        try:
            return OllamaResponse(**raw)
        except ValidationError as e:
            bad_fields = [err["loc"][0] for err in e.errors()]
            raise ValueError(
                f"Ollama response failed validation on field(s): {bad_fields}"
            ) from e

    def chat(self, messages: list[dict]) -> str:
        """Send a chat-style message history to the local Ollama model."""
        response = ollama.chat(model=self.model, messages=messages)
        return response["message"]["content"]


class ChatSession:
    """Stateful wrapper around ``OllamaClient.chat`` for multi-turn chats."""

    def __init__(self, client: OllamaClient):
        self.client = client
        self.messages: list[dict] = []

    def send(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        reply = self.client.chat(self.messages)
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def run(self):
        print("Chat started. Type 'exit' or 'quit' to stop.")
        while True:
            user_input = input("You: ")
            if user_input.lower() in ("exit", "quit"):
                break
            reply = self.send(user_input)
            print(f"Bot: {reply}")
