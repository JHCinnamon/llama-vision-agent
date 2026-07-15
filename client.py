"""Small command-line client for the local LLaVA agent API."""

from __future__ import annotations

import argparse

import requests
from rich import print


class LLaVAClient:
    def __init__(self, api_url: str) -> None:
        self.api_url = api_url

    def build_payload(self, prompt: str) -> dict[str, str]:
        return {"prompt": prompt}

    def send_request(self, prompt: str) -> dict[str, str]:
        response = requests.post(self.api_url, json=self.build_payload(prompt), timeout=60)
        response.raise_for_status()
        return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="LLaVA agent client")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to send to the local agent")
    parser.add_argument("--port", type=int, default=8002, help="Port number of the API server")
    args = parser.parse_args()

    client = LLaVAClient(f"http://127.0.0.1:{args.port}/generate")
    response = client.send_request(args.prompt)
    print(response)


if __name__ == "__main__":
    main()
