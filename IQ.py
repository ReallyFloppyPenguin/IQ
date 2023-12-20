import pyglet
import time
from gtts import gTTS
import json
import string
import random
import nltk
import numpy as num
from nltk.stem import WordNetLemmatizer  # It has the ability to lemmatize.
import os
import halo
import spinners
import utils.funcs
import speech_recognition as sr
import winsound
import pyttsx3

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf  # A multidimensional array of elements is represented by this symbol.
from keras import Sequential
from keras.models import load_model

# Sequential groups a linear stack of layers into a tf.keras.Model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

# nltk.download("punkt")  # required package for tokenization
# nltk.download("wordnet")  # word database
is_using_defined_neuron_formation = False
is_using_saved_model = False
audio_mode = False
is_voice_male = False
model_destination = "models/model.v3.iq"
neuron_formation = [
    500,
    500,
    1000,
    1000,
    1000,
    10_000,
    10_000,
    10_000,
    1000,
    1000,
    1000,
    500,
    500,
]
tf_verbosity_level = "3"
dataset_path = "training_data/training_data.v4.iq.json"
language = "en"
random_state = 54
test_size = 0.25
epochs = 1000
batch_size = 1
validation_split = 0.5
learning_rate = 0.001
voice_rate = 150
understanding_threshold = 0.20
loading = halo.Halo("Loading IQ", spinner=spinners.Spinners.dots12.value)
callbacks = [EarlyStopping(patience=3)]


if audio_mode:
    try:
        mic = sr.Microphone(device_index=3)
    except:
        print("Sorry, looks like your mic isn't plugged in")
        audio_mode = False

engine = pyttsx3.init()
voices = engine.getProperty("voices")
if is_voice_male:
    v = voices[0].id
else:
    v = voices[1].id
engine.setProperty("voice", v)
engine.setProperty("rate", voice_rate)


def respond(msg):
    if audio_mode:
        try:
            os.remove("./talk.wav")
        except:
            pass
        try:
            msg = msg.replace("<IQ> ", "")
            voice = gTTS(text=msg, lang=language, slow=False)
            voice.save("talk.wav")
            sound = pyglet.media.load("talk.wav")
            sound.play()
            time.sleep(sound.duration)
        except OSError:
            # Failed to use google-text-to-speech, must use fallback voice
            engine.say(msg)
            engine.runAndWait()

    else:
        print(msg)


def listen(mic, recognizer):
    with mic as source:
        r.adjust_for_ambient_noise(source)
        print("IQ voice recognition online (Talk!)")
        audio = recognizer.listen(source)
    return recognizer.recognize(audio)


if is_using_saved_model:
    loading.start()
lm = WordNetLemmatizer()  # returns the non-plural version of the word
newWords = []
documentX = []
documentY = []
classes = []
training_data = json.load(open(dataset_path))
# Each intent is tokenized into words and the patterns and their associated tags are added to their respective lists.
for intent in training_data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)  # tokenize the patterns
        newWords.extend(tokens)  # extends the tokens
        # print(tokens, "t")
        # print(newWords, "nw")
        documentX.append(pattern)
        documentY.append(intent["tag"])

    if intent["tag"] not in classes:  # add unexisting tags to their respective classes
        classes.append(intent["tag"])
# print(documentX)

newWords = [
    lm.lemmatize(word.lower()) for word in newWords if word not in string.punctuation
]  # set words to lowercase if not in punctuation (as . is punctuation)
newWords = sorted(set(newWords))  # sorting words
ourClasses = sorted(set(classes))  # sorting classes
trainingData = []  # training list array
outEmpty = [0] * len(ourClasses)


for idx, doc in enumerate(documentX):
    bagOfwords = []
    text = lm.lemmatize(doc.lower())
    # print(text)
    for word in newWords:
        bagOfwords.append(1) if word in text else bagOfwords.append(0)

    outputRow = list(outEmpty)
    outputRow[ourClasses.index(documentY[idx])] = 1
    trainingData.append([bagOfwords, outputRow])
    # print(bagOfwords)
    # print(trainingData)
    # print(outputRow)
    random.shuffle(trainingData)
trainingData = num.array(
    trainingData, dtype=object
)  # coverting our data into an array afterv shuffling

x = num.array(list(trainingData[:, 0]))  # first trainig phase
y = num.array(list(trainingData[:, 1]))  # second training phase
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=random_state, test_size=test_size
)
iShape = (len(x[0]),)
oShape = len(y[0])

if not is_using_saved_model:
    # parameter definition
    model = Sequential()
    # In the case of a simple stack of layers, a Sequential model is appropriate

    if not is_using_defined_neuron_formation:
        # Dense function adds an output layer
        model.add(Dense(128, input_shape=iShape, activation="relu"))
        # The activation function in a neural network is in charge of converting the node's summed weighted input into activation of the node or output for the input in question
        model.add(Dropout(0.5))
        # Dropout is used to enhance visual perception of input neurons
        model.add(Dense(500, activation="relu"))
        model.add(Dense(500, activation="relu"))
        model.add(Dense(1000, activation="relu"))
        model.add(Dense(1000, activation="relu"))
        model.add(Dense(1000, activation="relu"))
        model.add(Dense(10_000, activation="relu"))
        model.add(Dense(1000, activation="relu"))
        model.add(Dense(1000, activation="relu"))
        model.add(Dense(1000, activation="relu"))
        model.add(Dense(500, activation="relu"))
        model.add(Dense(500, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(oShape, activation="softmax"))
    else:
        model.add(Dense(128, input_shape=iShape, activation="relu"))
        # The activation function in a neural network is in charge of converting the node's summed weighted input into activation of the node or output for the input in question
        model.add(Dropout(0.5))
        for neuron in neuron_formation:
            model.add(Dense(neuron, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(oShape, activation="softmax"))
    # below is a callable that returns the value to be used with no arguments
    md = tf.keras.optimizers.Adam(learning_rate=learning_rate, decay=1e-6)
    # Below line improves the numerical stability and pushes the computation of the probability distribution into the categorical crossentropy loss function.
    model.compile(loss="categorical_crossentropy", optimizer=md, metrics=["accuracy"])
    # Output the model in summary
    print(model.summary())
    # Whilst training your Nural Network, you have the option of making the output verbose or simple.
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1,
    )
    model.save(model_destination)
    winsound.PlaySound("SystemExclamation", winsound.SND_ALIAS)


def ourText(text):
    newtkns = nltk.word_tokenize(text)
    newtkns = [lm.lemmatize(word) for word in newtkns]
    return newtkns


def wordBag(text, vocab):
    tokens = ourText(text)
    bagOwords = [0] * len(vocab)
    for w in tokens:
        for idx, word in enumerate(vocab):
            if word == w:
                bagOwords[idx] = 1
    return num.array(bagOwords)


model = load_model(model_destination)
if is_using_saved_model:
    loading.stop()


def Pclass(text, vocab, labels):
    bagOwords = wordBag(text, vocab)
    ourResult = model.predict(num.array([bagOwords]), verbose=0)[0]
    # print(ourResult)
    yp = [
        [idx, res]
        for idx, res in enumerate(ourResult)
        if res >= understanding_threshold
    ]
    # print(yp, "yp")
    yp.sort(key=lambda x: x[1], reverse=True)
    if len(yp) == 0:
        return []
    newList = []
    for r in yp:
        # print(r, "r")
        newList.append(labels[r[0]])
    return newList


def exec_res(firstlist, fJson, message):
    tag = firstlist[0]
    listOfIntents = fJson["intents"]
    for i in listOfIntents:
        if i["tag"] == tag:
            res = random.choice(i["responses"])
            exec(f'utils.funcs.{res}("{message}", respond, listen)', globals())
            break


if audio_mode:
    r = sr.Recognizer()
    # r.energy_threshold = 1038
    r.dynamic_energy_threshold = True
    mic = sr.Microphone(device_index=3)
while True:
    if audio_mode:
        try:
            human_req = listen(mic, r)
        except LookupError:
            respond("<IQ> Sorry, i didn't get that, please try again")
            continue
    else:
        human_req = input("<HOOMAN> ")
    intents = Pclass(human_req, newWords, ourClasses)
    if len(intents) == 0:
        respond("Sorry, i don't know what you mean.")
        continue
    exec_res(intents, training_data, human_req)
