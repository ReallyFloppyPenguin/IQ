import json
import socket
import time
import webbrowser
from ipregistry import IpregistryClient
import requests


def lookup(message, voice, talk):
    if talk:
        voice.speak("ok")
    webbrowser.Chrome("C:/Program Files/Google/Chrome/Application/chrome.exe").open(
        f"""https://www.google.com/search?q={"+".join(message
                                      .replace("lookup", "")
                                      .replace("google", "")
                                      .split(" "))}"""
    )


def telljoke(message, voice, talk):
    if talk:
        voice.speak("what does a joke do?")
        time.sleep(1)
        voice.speak("it makes laughs!")
    else:
        print("what does a joke do?")
        time.sleep(1)
        print("it makes laughs!")


def finddevice(message, voice, talk):
    requests.post(
        "https://ntfy.sh/iqassistant1234",
        data="Hi!".encode(encoding="utf-8"),
        headers={
            "Title": "Tryna find ur device",
            "Priority": "urgent",
        },
    )


def getpassword(message, voice, talk):
    passwords = json.load(open("password.json"))
    for pw in passwords.keys():
        if pw in message:
            password = passwords[pw]
            requests.post(
                "https://ntfy.sh/iqassistant1234",
                data=f"Your password is {password}".encode(encoding="utf-8"),
                headers={
                    "Title": "here's your password",
                },
            )
            return
    if talk:
        voice.speak("could not find your password")
    print("could not find your password")


def getweather(x, y, z):
    import requests, json

    # Enter your API key here
    api_key = "201be1e581e80f66248e1cf4dc64dc81"

    # base_url variable to store url
    base_url = "https://api.openweathermap.org/data/2.5/weather?"
    # Give city name
    city_name = "Milton Keynes"

    # complete_url variable to store
    # complete url address
    complete_url = base_url + "appid=" + api_key + "&q=" + city_name

    # get method of requests module
    # return response object
    response = requests.get(complete_url)

    # json method of response object
    # convert json format data into
    # python format data
    x = response.json()
    open("response.json", "x").write(json.dumps(x))
    # Now x contains list of nested dictionaries
    # Check the value of "cod" key is equal to
    # "404", means city is found otherwise,
    # city is not found
    if x["cod"] != "404":

        # store the value of "main"
        # key in variable y
        y = x["main"]

        # store the value corresponding
        # to the "temp" key of y
        current_temperature = y["temp"]

        # store the value corresponding
        # to the "pressure" key of y
        current_pressure = y["pressure"]

        # store the value corresponding
        # to the "humidity" key of y
        current_humidity = y["humidity"]

        # store the value of "weather"
        # key in variable z
        z = x["weather"]

        # store the value corresponding
        # to the "description" key at
        # the 0th index of z
        weather_description = z[0]["description"]

        # print following values
        print(
            " Temperature (celcius) = "
            + str(current_temperature - 273.15)
            + "\n atmospheric pressure (in hPa unit) = "
            + str(current_pressure)
            + "\n humidity (in percentage) = "
            + str(current_humidity)
            + "\n description = "
            + str(weather_description)
        )
    else:
        print(" City Not Found ")


getweather(0, 1, 2)
