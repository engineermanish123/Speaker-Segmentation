README: Speaker Segmentation & Diarization Script

Overview

This script performs speaker segmentation and diarization on audio files. It processes audio recordings, detects different speakers, and saves the segmented audio files for each detected speaker.

Features

Loads audio files from the processed_audio/ folder.

Performs speaker diarization using pyAudioAnalysis.

Normalizes the audio to improve processing.

Segments speakers and saves individual speaker audio files in separated_speakers/.

Handles empty or invalid audio files gracefully.

Dependencies

Ensure the following Python libraries are installed:

pip install numpy librosa soundfile pyAudioAnalysis

Folder Structure

processed_audio/ → Folder containing input audio files (only .wav files are supported).

separated_speakers/ → Output folder where segmented speaker files are saved.

requirements.txt → List of required dependencies.

How to Run the Script

Place .wav files in the processed_audio/ folder.

Run the script using:

python script_name.py

Processed files will be saved in separated_speakers/.

Notes

The script only works with WAV files.

You can adjust the number of expected speakers (n_speakers=3) in the script.

If no audio files are found, the script will exit with a message.

For any issues, please contact the development team.

