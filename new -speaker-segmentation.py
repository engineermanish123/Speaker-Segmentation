import os
import numpy as np
import librosa
import soundfile as sf
import torchaudio
import subprocess
from pyAudioAnalysis import audioSegmentation as aS
                            
# Set your FFmpeg path
FFMPEG_PATH = r"C:\Users\user4\Downloads\ffmpeg\bin\ffmpeg.exe"

# Input and output folders
input_folder = "processed_audio"
output_folder = "separated_speakers"
temp_folder = "temp_audio"
os.makedirs(output_folder, exist_ok=True)
os.makedirs(temp_folder, exist_ok=True)

# Allowed audio formats
allowed_formats = (".wav", ".mp3", ".flac", ".aac", ".m4a", ".ogg")

# Get all audio files
audio_files = [f for f in os.listdir(input_folder) if f.lower().endswith(allowed_formats)]

if not audio_files:
    print(" No audio files found in 'processed_audio' folder!")
    exit()

for file in audio_files:
    file_path = os.path.join(input_folder, file)
    converted_file_path = file_path

    # Convert non-WAV files to WAV using FFmpeg
    if not file.lower().endswith(".wav"):
        converted_file_path = os.path.join(temp_folder, os.path.splitext(file)[0] + ".wav")
        print(f" Converting {file} to WAV...")
        try:
            subprocess.run(
                [FFMPEG_PATH, "-i", file_path, "-ar", "16000", "-ac", "1", converted_file_path, "-y"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True
            )
            print(f" Converted: {converted_file_path}")
        except subprocess.CalledProcessError as e:
            print(f" FFmpeg conversion failed for '{file}': {str(e)}")
            continue

    # Load audio with torchaudio
    print(f" Loading: {converted_file_path}")
    try:
        waveform, sr = torchaudio.load(converted_file_path)
        y = waveform.mean(dim=0).numpy()  # Convert to NumPy array (Mono)
    except Exception as e:
        print(f" Error loading '{file}' with torchaudio: {e}")
        continue

    # Normalize audio
    y = librosa.util.normalize(y)

    # Perform speaker diarization
    print(f" Running speaker diarization on {file}...")
    try:
        segments, _, _ = aS.speaker_diarization(converted_file_path, n_speakers=3)
    except Exception as e:
        print(f" Speaker diarization failed for '{file}': {str(e)}")
        continue

    if segments is None or len(segments) == 0:
        print(f" Error: Speaker diarization failed for '{file}'.")
        continue

    segments = segments.astype(int)

    # Extract speaker-wise segments
    speaker_waves = {}
    segment_duration = len(y) / len(segments)
    min_samples = sr * 10  # 10 seconds in samples
    max_samples = sr * 15  # 15 seconds in samples

    for i, speaker_id in enumerate(segments):
        start_sample = int(i * segment_duration)
        end_sample = int((i + 1) * segment_duration)

        if speaker_id not in speaker_waves:
            speaker_waves[speaker_id] = []

        speaker_waves[speaker_id].append(y[start_sample:end_sample])

    # Save speaker-wise audio files
    for speaker, wave_chunks in speaker_waves.items():
        speaker_audio = np.concatenate(wave_chunks)
        duration = len(speaker_audio)

        if duration >= min_samples:
            speaker_audio = speaker_audio[:max_samples]  # Trim to max 15 sec

        output_file = os.path.join(output_folder, f"{os.path.splitext(file)[0]}_Speaker_{speaker}.wav")
        sf.write(output_file, speaker_audio, sr)
        print(f" Saved: {output_file}")

print(" Speaker segmentation completed successfully!")
