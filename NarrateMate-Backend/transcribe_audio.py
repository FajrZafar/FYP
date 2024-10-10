
import subprocess
import speech_recognition as sr
import sys
import os

def convert_to_wav(input_file, output_file):
    try:
        subprocess.run([
            'ffmpeg', '-i', input_file, output_file
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting file: {e}", file=sys.stderr)
        raise

def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            return text
    except Exception as e:
        print(f"Error during transcription: {e}", file=sys.stderr)
        raise

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python transcribe_audio.py <audio_file>", file=sys.stderr)
        sys.exit(1)

    input_file = sys.argv[1]
    base, ext = os.path.splitext(input_file)
    wav_file = base + '.wav'

    if ext.lower() != '.wav':
        try:
            # Convert to WAV if not already in WAV format
            convert_to_wav(input_file, wav_file)
            audio_file = wav_file
        except Exception as e:
            print(f"Conversion failed: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        audio_file = input_file

    try:
        transcription = transcribe_audio(audio_file)
        print(transcription)
    except Exception as e:
        print(f"Transcription failed: {e}", file=sys.stderr)
        sys.exit(1)

    if wav_file != input_file and os.path.exists(wav_file):
        os.remove(wav_file)

