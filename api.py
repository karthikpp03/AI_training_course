from flask import Flask, jsonify, request
import requests


app = Flask(__name__)

API_KEY = "542d71d13e0434051d3948d9cbb805a7"
BASE_URL="http://api.openweathermap.org/data/2.5/weather"

@app.route("/test")

def weather():
    city =request.args.get("city","Avadi")

    params ={
        "q":city,
        "appid": API_KEY

    }
    print(params)
    response = requests.get(BASE_URL,params=params)
    data =response.json()

    if data.get("cod") !=200:
        return jsonify({"error":data.get("message","something wrong")}),400
    result = {
        "city":data["name"],
        "temperature":data["main"]["temp"],
        "weather":data["weather"][0]["description"]

    }
    return jsonify(result)        

if __name__ == "__main__":
    app.run(debug=True)