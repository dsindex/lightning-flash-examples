import requests

text = "Best movie ever"
body = {"session": "UUID", "payload": {"inputs": {"data": text}}}
resp = requests.post("http://127.0.0.1:8000/predict", json=body)

print(resp.json())
