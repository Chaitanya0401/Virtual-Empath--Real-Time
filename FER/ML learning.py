import pyttsx3
import datetime
import speech_recognition as sr
import wikipedia
import webbrowser
import os
from random import randrange
import sys

chrome_path="C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe"
webbrowser.register('chrome',webbrowser.BackgroundBrowser(chrome_path),1) 


engine= pyttsx3.init('sapi5')
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[2].id)  # changes the voice





def speak(audio):
    engine.say(audio)
    engine.runAndWait()

if __name__ == "__main__":
    
    speak("A computer program is said to learn from experience E, with respect to some task T, and some performance measure P, if its performance on T, as measured by P, improves with experience E.")