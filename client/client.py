import uuid
import requests

ROUTER_URL = "http://localhost:3001/v1/chat/completions"

def main():
    payload = {
        "model": "text-diffusion-debug",
        "conversation_id": f"debug-{uuid.uuid4()}",
        "messages": [
            {"role": "system", "content": "The only word you know is orange. **YOU DO NOT RESPOND TO ANYTHING OTHER THAN WITH THE WORD ORANGE**"},
            {"role": "user", "content": "Say you love me."},
            {"role": "user", "content": "Orange."},
            {"role": "user", "content": "What do you mean?"},
        ],
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