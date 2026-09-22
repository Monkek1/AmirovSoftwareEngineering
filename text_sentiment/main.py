from transformers import pipeline

# https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

'''
result = (pipe("I hate you"))[0]

print(result["label"], result["score"])
'''

def sentiment_func(text):
    result = pipe(text)[0]

    return {
        "label": result["label"],
        "score": result["score"]
    }