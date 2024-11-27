# voice_assistant/audio.py

import speech_recognition as sr
import pygame
import time
import logging
import pydub
from io import BytesIO
from pydub import AudioSegment
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@lru_cache(maxsize=None)
def get_recognizer():
    """
    Return a cached speech recognizer instance
    """
    return sr.Recognizer()

def record_audio(file_path, stop_event=None, timeout=10, phrase_time_limit=None, 
                 energy_threshold=2000, pause_threshold=1, phrase_threshold=0.1, 
                 dynamic_energy_threshold=True, calibration_duration=1):
    """
    Record audio from the microphone and save it as an MP3 file.
    """
    recognizer = get_recognizer()
    recognizer.energy_threshold = energy_threshold
    recognizer.pause_threshold = pause_threshold
    recognizer.phrase_threshold = phrase_threshold
    recognizer.dynamic_energy_threshold = dynamic_energy_threshold

    try:
        with sr.Microphone() as source:
            logging.info("Calibrating for ambient noise...")
            start_time = time.time()
            while time.time() - start_time < calibration_duration:
                if stop_event and stop_event.is_set():
                    logging.info("Stop event detected during calibration. Exiting.")
                    return "stopped"
                recognizer.adjust_for_ambient_noise(source, duration=0.1)

            if stop_event and stop_event.is_set():
                logging.info("Stop event detected after calibration. Exiting.")
                return "stopped"

            logging.info("Recording started.")
            audio_data = None
            start_time = time.time()
            while True:
                if stop_event and stop_event.is_set():
                    logging.info("Stop event detected during recording. Exiting.")
                    return "stopped"
                try:
                    audio_data = recognizer.listen(source, timeout=0.5, phrase_time_limit=phrase_time_limit)
                    break
                except sr.WaitTimeoutError:
                    if time.time() - start_time > timeout:
                        logging.warning("Recording timeout exceeded.")
                        return None

            logging.info("Recording complete.")
            wav_data = audio_data.get_wav_data()
            audio_segment = pydub.AudioSegment.from_wav(BytesIO(wav_data))
            audio_segment.export(file_path, format="mp3", bitrate="128k", parameters=["-ar", "22050", "-ac", "1"])
            return file_path

    except Exception as e:
        logging.error(f"Failed to record audio: {e}")
        return None



def play_audio(file_path, stop_event=None):
    """
    Play an audio file using pygame.
    
    Args:
    file_path (str): The path to the audio file to play.
    stop_event: Event to stop playback.
    """
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            if stop_event and stop_event.is_set():
                logging.info("Playback stopped by user.")
                pygame.mixer.music.stop()
                break
            pygame.time.wait(100)
    except pygame.error as e:
        logging.error(f"Failed to play audio: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred while playing audio: {e}")
    finally:
        pygame.mixer.quit()
