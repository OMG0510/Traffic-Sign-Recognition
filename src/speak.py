import win32com.client
import threading

speaker = win32com.client.Dispatch("SAPI.SpVoice")

def speak(text):
    def run():
        speaker.Speak(text)
    t = threading.Thread(target=run)
    t.daemon = True
    t.start()