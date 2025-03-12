import os
import numpy as np
import librosa
import soundfile as sf
from pyAudioAnalysis import audioSegmentation as aS


input_folder = "processed_audio"
output_folder = "separated_speakers"
os.makedirs(output_folder, exist_ok=True)


audio_files = [f for f in os.listdir(input_folder) if f.endswith(".wav")]

if not audio_files:
    print(" No audio files found in 'processed_audio' folder!")
    exit()

for file in audio_files:
    file_path = os.path.join(input_folder, file)

    
    print(f" Loading: {file_path}")
    try:
        y, sr = librosa.load(file_path, sr=None)
        if y is None or len(y) == 0:
            print(f" Error: Audio file '{file}' is empty.")
            continue
    except Exception as e:
        print(f" Error loading '{file}': {str(e)}")
        continue

    
    y = librosa.util.normalize(y)

    
    print(f" Running speaker diarization on {file}...")
    segments, _, _ = aS.speaker_diarization(file_path, n_speakers=3)  

    if segments is None or len(segments) == 0:
        print(f" Error: Speaker diarization failed for '{file}'.")
        continue

    segments = segments.astype(int)

   
    speaker_waves = {}
    segment_duration = len(y) / len(segments)
    min_samples = sr * 10 
    max_samples = sr * 15  

    for i, speaker_id in enumerate(segments):
        start_sample = int(i * segment_duration)
        end_sample = int((i + 1) * segment_duration)

        if speaker_id not in speaker_waves:
            speaker_waves[speaker_id] = []
        speaker_waves[speaker_id].append(y[start_sample:end_sample])
    
    for speaker, wave_chunks in speaker_waves.items():
        speaker_audio = np.concatenate(wave_chunks)
        duration = len(speaker_audio)

        if duration >= min_samples:
            
            speaker_audio = speaker_audio[:max_samples]  
        else:
           
            pass  

        output_file = os.path.join(output_folder, f"{file}_Speaker_{speaker}.wav")
        sf.write(output_file, speaker_audio, sr)
        print(f" Saved: {output_file}")

print(" Speaker segmentation completed successfully!")
