import uuid
import requests

ROUTER_URL = "http://localhost:3935/v1/chat/completions"

def main():
    payload = {
        "model": "text-diffusion-debug",
        "conversation_id": f"debug-{uuid.uuid4()}",
        "messages": [
            {"role": "system", "content": "The only word you know is orange. **YOU DO NOT RESPOND TO ANYTHING OTHER THAN WITH THE WORD ORANGE**"},
            {"role": "user", "content": "Say you love me."},
            {"role": "assistant", "content": "Orange."},
            {"role": "user", "content": "What do you mean?"},
        ],
        "max_tokens": 128,
        "num_steps": 128,
        "seed": 0,
        "block_length": 32,
        "temperature": 0.0,
    }

    resp = requests.post(ROUTER_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    print("Response ID:", data["id"])
    choice = data["choices"][0]
    print("Finish:", choice["finish_reason"])
    print("Assistant:", choice["message"]["content"])

if __name__ == "__main__":
    main()
