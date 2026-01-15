# app/voice.py
import pyttsx3
import threading
import time


def _speak_worker(text: str):
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 150)

        # 🔑 IMPORTANT: speak everything in ONE call
        engine.say(text)
        engine.runAndWait()

        engine.stop()

    except Exception:
        # ABSOLUTE SILENT FAIL (Pi / Bluetooth safe)
        pass


def speak(text: str):
    """
    Non-blocking, safe TTS.
    Works on PC + Raspberry Pi.
    """
    if not text or not text.strip():
        return

    threading.Thread(
        target=_speak_worker,
        args=(text,),
        daemon=True
    ).start()