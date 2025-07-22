from faster_whisper import WhisperModel

# Load the model with GPU acceleration
model = WhisperModel("shunyalabs/pingala-v1-en-verbatim", device="cuda", compute_type="float16")

# Transcribe English audio
segments, info = model.transcribe("audio.wav", beam_size=5)

# Print transcription results
for segment in segments:
    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
