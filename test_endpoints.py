import httpx
import sys

base_url = "http://127.0.0.1:8000"

try:
    # Test Root
    print("Testing Root...")
    resp = httpx.get(f"{base_url}/")
    print(f"Root: {resp.status_code} {resp.json()}")

    # Test Favicon
    print("Testing Favicon...")
    resp = httpx.get(f"{base_url}/favicon.ico")
    print(f"Favicon: {resp.status_code} (Expect 204)")

    # Test Chat
    print("Testing Chat...")
    resp = httpx.post(f"{base_url}/chat", json={"message": "Hello"})
    print(f"Chat: {resp.status_code} {resp.json()}")

except Exception as e:
    print(f"Error: {e}")
