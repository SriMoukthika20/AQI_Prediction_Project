import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "PM2.5": 45,
    "PM10": 80,
    "NO2": 30,
    "SO2": 12,
    "CO": 1.2,
    "O3": 40
}

response = requests.post(url, json=data)

print(response.json())