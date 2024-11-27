import speech_recognition as sr
recognizer = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    audio = recognizer.listen(source)
    print("Got it!")
