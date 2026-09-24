from transformers import pipeline

# https://huggingface.co/docs/transformers/tasks/asr

pipe = pipeline(task="automatic-speech-recognition", model="openai/whisper-base")
print(pipe("audio_stt/samples/audio1.wav"))