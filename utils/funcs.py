from data import *
import os
import timer
import timeit
from threading import Timer

def greeting(message, respond, listen):
    respond(f"Hi")


runhi = greeting


def runname(message, respond, listen):
    respond(f"My name is {name}")


def goodbye(message, respond, listen):
    respond(f"Bye-Bye")


def runage(message, respond, listen):
    respond(f"My age is {age}")


def runexe(message, respond, listen):
    message = message.lower()
    keyword = "open"
    before, keyword, after = message.partition(keyword)
    after = after.strip()
    if after in ["rl", "rocket league"]:
        os.startfile(r"C:\Users\Admin\OneDrive\Desktop\Rocket League®.url")
        respond("Opening RL")
    elif after in ["calc", "calculator"]:
        os.startfile(r"C:\\Windows\\System32\\calc.exe")
        respond("Opening Calculator")
    elif after in ["notepad"]:
        os.startfile(r"C:\\Windows\\System32\\notepad.exe")
        respond("Opening Notepad")
    respond("Sorry, i don't know what to open")


def starttimer(message: str, respond, listen):
    message = message.lower()
    minutes, m_keyword, _ = message.partition("minutes")
    seconds, s_keyword, _ = message.partition("seconds")
    minutes = minutes.strip()
    print(minutes, seconds)
    respond(f"{minutes} minutes on the timer")
    if s_keyword:
        seconds = (int(minutes) * 60) + int(seconds.strip())
    else:
        seconds = int(minutes) * 60
    t = Timer(int(minutes), lambda: respond("Your timer is over"))
    t.start()

