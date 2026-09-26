from transformers import pipeline

# https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics

pipe = pipeline(task="video-classification", model="MCG-NJU/videomae-base-finetuned-kinetics")
print(pipe("video_detection/samples/testvideo.mp4"))