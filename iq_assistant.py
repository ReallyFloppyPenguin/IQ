import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import string
import random
import nltk
import numpy as num
from nltk.stem import WordNetLemmatizer  # It has the ability to lemmatize.
import tensorflow as tf  # A multidimensional array of elements is represented by this symbol.

tf.get_logger().setLevel("ERROR")
from keras import (
    Sequential,
)  # Sequential groups a linear stack of layers into a tf.keras.Model
from keras.models import load_model
from keras.layers import Dense, Dropout
import iq_functions
import speech_recognition as sr
import pyttsx3


nltk.download("punkt")  # required package for tokenization
nltk.download("wordnet")  # word database
lm = WordNetLemmatizer()  # for getting words
engine = pyttsx3.init()
ourClasses = []
newWords = []
documentX = []
documentY = []
memory = {}
is_training = False
audio_mode = False
talk = True
ourData = {
    "ourIntents": [
        {
            "tag": "lookup",
            "patterns": ["lookup python dictionary", "google how to uninstall windows"],
            "responses": iq_functions.lookup,
        },
        {
            "tag": "telljoke",
            "patterns": [
                "tell me a joke",
                "im feeling funny",
                "give me your best punchlines",
            ],
            "responses": iq_functions.telljoke,
        },
        {
            "tag": "finddevice",
            "patterns": ["find mvs device", "find gps device", "ive lost my device"],
            "responses": iq_functions.finddevice,
        },
        {
            "tag": "getpassword",
            "patterns": [
                "send me my google password",
                "get me my epic password",
                "can i have my lego password",
            ],
            "responses": iq_functions.getpassword,
        },
    ]
}


def listen(mic, recognizer):
    with mic as source:
        r.adjust_for_ambient_noise(source)
        print("IQ voice recognition online (Talk!)")
        audio = recognizer.listen(source)
    return recognizer.recognize(audio)

def respond(response):
    if talk:
        pyttsx3.speak(response)
    else:
        print(response)

for intent in ourData["ourIntents"]:
    for pattern in intent["patterns"]:
        ournewTkns = nltk.word_tokenize(pattern)  # tokenize the patterns
        newWords.extend(ournewTkns)  # extends the tokens
        documentX.append(pattern)
        documentY.append(intent["tag"])

    if (
        intent["tag"] not in ourClasses
    ):  # add unexisting tags to their respective classes
        ourClasses.append(intent["tag"])

newWords = [
    lm.lemmatize(word.lower()) for word in newWords if word not in string.punctuation
]  # set words to lowercase if not in punctuation
newWords = sorted(set(newWords))  # sorting words
ourClasses = sorted(set(ourClasses))  # sorting classes
trainingData = []  # training list array
outEmpty = [0] * len(ourClasses)
# bow model
for idx, doc in enumerate(documentX):
    bagOfwords = []
    text = lm.lemmatize(doc.lower())
    for word in newWords:
        bagOfwords.append(1) if word in text else bagOfwords.append(0)

    outputRow = list(outEmpty)
    outputRow[ourClasses.index(documentY[idx])] = 1
    trainingData.append([bagOfwords, outputRow])

random.shuffle(trainingData)
trainingData = num.array(
    trainingData, dtype=object
)  # coverting our data into an array afterv shuffling
x = num.array(list(trainingData[:, 0]))  # first trainig phase
y = num.array(list(trainingData[:, 1]))  # second training phase
iShape = (len(x[0]),)
oShape = len(y[0])
if is_training:
    # Each intent is tokenized into words and the patterns and their associated tags are added to their respective lists.
    # parameter definition
    ourNewModel = Sequential()
    # In the case of a simple stack of layers, a Sequential model is appropriate

    # Dense function adds an output layer
    ourNewModel.add(Dense(128, input_shape=iShape, activation="relu"))
    # The activation function in a neural network is in charge of converting the node's summed weighted input into activation of the node or output for the input in question
    ourNewModel.add(Dropout(0.5))
    # Dropout is used to enhance visual perception of input neurons
    ourNewModel.add(Dense(64, activation="relu"))
    ourNewModel.add(Dropout(0.3))
    ourNewModel.add(Dense(oShape, activation="softmax"))
    # below is a callable that returns the value to be used with no arguments
    md = tf.keras.optimizers.Adam(learning_rate=0.01, decay=1e-6)
    # Below line improves the numerical stability and pushes the computation of the probability distribution into the categorical crossentropy loss function.
    ourNewModel.compile(
        loss="categorical_crossentropy", optimizer=md, metrics=["accuracy"]
    )
    # Output the model in summary
    print(ourNewModel.summary())
    # Whilst training your Nural Network, you have the option of making the output verbose or simple.
    ourNewModel.fit(x, y, epochs=200, verbose=1)
    ourNewModel.save("models\model.v8.stable.iq")

ourNewModel = load_model("models\model.v8.stable.iq")


def ourText(text):
    newtkns = nltk.word_tokenize(text)
    newtkns = [lm.lemmatize(word) for word in newtkns]
    return newtkns


def wordBag(text, vocab):
    newtkns = ourText(text)
    bagOwords = [0] * len(vocab)
    for w in newtkns:
        for idx, word in enumerate(vocab):
            if word == w:
                bagOwords[idx] = 1
    return num.array(bagOwords)


def Pclass(text, vocab, labels):
    bagOwords = wordBag(text, vocab)
    ourResult = ourNewModel.predict(num.array([bagOwords]))[0]
    newThresh = 0.2
    yp = [[idx, res] for idx, res in enumerate(ourResult) if res > newThresh]

    yp.sort(key=lambda x: x[1], reverse=True)
    newList = []
    for r in yp:
        newList.append(labels[r[0]])
    return newList


def getRes(firstlist, fJson, message):
    tag = firstlist[0]
    listOfIntents = fJson["ourIntents"]
    for i in listOfIntents:
        if i["tag"] == tag:
            memory[message] = i["responses"]
            i["responses"](message, respond)


if audio_mode == True:
    r = sr.Recognizer()
    # r.energy_threshold = 1038
    r.dynamic_energy_threshold = True
    mic = sr.Microphone()
while True:
    if audio_mode == True:
        try:
            human_req = listen(mic, r)
        except LookupError:
            continue
    else:
        human_req = input("<HOOMAN> ")
    intents = Pclass(human_req, newWords, ourClasses)
    getRes(intents, ourData, human_req)
#  me ⅏⅏⅏⅏⅏⅏ you