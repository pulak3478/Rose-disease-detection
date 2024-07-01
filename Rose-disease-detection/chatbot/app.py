from pyngrok import ngrok
from flask import Flask
import nltk
nltk.download('popular')
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from flask import Flask, render_template, request
from contextlib import redirect_stdout
from io import StringIO
import nltk
ngrok.set_auth_token("2eYSEkdHoXVvRNCoDM8OVWi7EBg_4GpSbsMrcbj72Jm6Cs6qh")
public_url=ngrok.connect(5001).public_url
port_no=5001

app = Flask(__name__)
ngrok.set_auth_token("2eYSEkdHoXVvRNCoDM8OVWi7EBg_4GpSbsMrcbj72Jm6Cs6qh")
app.static_folder = 'static'

lemmatizer = WordNetLemmatizer()
model = load_model('model.h5')

intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)


def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']

    for intent in list_of_intents:
        if intent['tag'] == tag:
            responses = intent['responses']

            if 'response_with_links' in intent:
                links_response = intent['response_with_links']
                result = random.choice(responses)
                link_texts = []
                for link in links_response:
                    link_texts.append(f"{link['message']}<a href='{link['link']}' target='blank'>{link['text']}</a>")
                return {"text": result + " ".join(link_texts), "link": True}
            else:
                result = random.choice(responses)
                return {"text": result, "link": False}

    return {"text": "I'm sorry, I don't understand that.", "link": False}

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)

    response_text = res["text"]
    response_link = res.get("link",'')
    link_text = res.get("link_text", "")

    if response_link:
        return f"{response_text} <a href='{link_text}' target='_blank'></a>"
    else:
        return response_text



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    responses = chatbot_response(userText)

    return json.dumps(responses)
print(f"To acces the Gloable link please click\n{public_url}")
if __name__ == "__main__":
    app.run(port=5001)
