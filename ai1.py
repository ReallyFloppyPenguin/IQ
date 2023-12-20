import numpy as np  # python package for arrays and operations
import pandas as pd  # python package for data processing
import re
import os
import time
import speech_recognition as sr
import pyttsx3 as voice
import halo
import spinners
from sklearn.model_selection import train_test_split

# Variables
is_using_saved_model = False
model_destination = "model.iq23434343434"
random_state = 54
test_size = 0.25
tf_verbosity_level = "3"
use_func_api = True  # EXPERIMENTAL
audio_mode = False
LSTM_dropout = 0.1
epochs = 100
batch_size = 1
validation_split = 0.2
Embedding_out_len = 20
is_voice_male = False
loading = halo.Halo("Loading IQ", spinner=spinners.Spinners.dots12.value)

if is_using_saved_model:
    loading.start()

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = tf_verbosity_level
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, Input, concatenate, Flatten

engine = voice.init()
voices = engine.getProperty("voices")
if is_voice_male:
    v = voices[0].id
else:
    v = voices[1].id
engine.setProperty("voice", v)
engine.setProperty("rate", 100)


def respond(msg):
    if audio_mode:
        engine.say(msg)
        engine.runAndWait()
    else:
        print(msg)


data = pd.read_csv("training_data.csv", encoding="ISO-8859-1")
data = data[["request", "action", "params"]]
data["request"] = data["request"].apply((lambda x: re.sub("[^a-zA-Z0-9\s]", "", x)))

# Change all letters of the words in each review to lower case
data["request"] = data["request"].apply(lambda x: x.lower())

max_words = 10000
tokenizer = Tokenizer(num_words=max_words, split=" ")
# create a vocabulary for the words in the texts assigning each #word a number. In Essence this creates a word index “dictionary”
tokenizer.fit_on_texts(data["request"].values)

# the following two lines show each word in the texts and the #number that tokenizer assigns it
words = tokenizer.word_index
# print(words)

# take each word in each text entry and replace it with its #corresponding integer value from the word_index dictionary. In
# other words represent each review as a series of numbers
X = tokenizer.texts_to_sequences(data["request"].values)
# Pads putting zeros from the beginning by default unless you #specify
X = pad_sequences(X)
# print(X, "X")


if not is_using_saved_model:
    if not use_func_api:
        model = Sequential()
        model.add(Embedding(max_words, 20, input_length=X.shape[1]))
        model.add(SpatialDropout1D(0.2))
        model.add(Dense(600, activation='relu'))
        model.add(LSTM(500, dropout=0.1))
        model.add(Dense(400, activation='relu'))
        model.add(Dense(4, activation="softmax"))
    else:
        # print(X.shape)
        input_layer = Input(shape=(X.shape[1],))
        print(input_layer.shape)
        emb = Embedding(max_words, Embedding_out_len, input_length=X.shape[1])(
            input_layer
        )
        x = SpatialDropout1D(0.1)(emb)
        x = Dense(400)(x)
        x = LSTM(500, dropout=LSTM_dropout)(x)
        # x = Flatten()(x)
        # x = Dense(600)(x)
        # LSTM(500, dropout=0.5)(input_layer)
        # LSTM(500, dropout=0.5)(input_layer)
        # LSTM(500, dropout=0.5)(input_layer)
        x = LSTM(500, dropout=LSTM_dropout)(x)
        x = Dense(400)(x)
        x = LSTM(300, dropout=LSTM_dropout)(x)
        x = Flatten()(x)
        x = Dense(200)(x)
        x = LSTM(100, dropout=LSTM_dropout)(x)
        x = Flatten()(x)
        # action = Dense(3, activation="softmax")(input_layer)
        # params = Dense(1, activation="softmax")(input_layer)
        # output_layer = concatenate([action, params])
        output_layer_2 = Dense(4, activation="softmax")(x)
        model = Model(inputs=input_layer, outputs=output_layer_2)
        model.summary()
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )

    # create a two column dummy variable using the data in the
    # sentiment column of the DataFrame named “data”
    # pd.get_dummies(data, columns=['actions'])

    # Assign the two column dummy variables to a numpy array #named Y

    # show what the dummy variables look like representing each #sentiment #(i.e. 1,0 means negative and 0,1 means positive)
    # show that Y has two columns

    action_dummy = pd.get_dummies(data["action"]).values
    Y = []
    for i, o in enumerate(action_dummy):
        # print(type(o))
        Y.append([*o, data["params"].values[i]])
    Y = np.array(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, random_state=random_state, test_size=test_size
    )

    # Show how many rows and how many columns are in the numpy arrays #X_train, Y_train, X_test and Y_test. Obtaining the “shape” means
    # obtaining the number of rows and number of columns of each array
    print(X_train.shape, Y_train.shape, "SHAPE_TRAIN")
    print(X_test.shape, Y_test.shape, "SHAPE_TEST")
    print(X[0])
    print(X.shape, "X_SHAPE")
    print(Y.shape, "Y_SHAPE")
    print(Y, "Y")

    model.fit(
        X_train,
        Y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        validation_split=validation_split,
    )
    score = model.evaluate(X_test, Y_test, verbose=0)
    print("Loss:", score[0])
    print("Accuracy:", score[1])
    model.save(model_destination)


# Test the neural networks ability to predict a sentiment by #creating #and entering a review using the lines below

# request = ['five mins']
model = load_model(model_destination)
r = sr.Recognizer()
# r.energy_threshold = 1038
r.dynamic_energy_threshold = True
mic = sr.Microphone(device_index=3)
# print(sr.Microphone.list_microphone_names())
if is_using_saved_model:
    loading.stop()

while True:
    # vectorizing the review by the pre-fitted tokenizer instance
    if audio_mode:
        try:
            with mic as source:
                r.adjust_for_ambient_noise(source)
                print("IQ voice recognition online (Talk!)")
                audio = r.listen(source)
            human_req = r.recognize(audio)
            print(human_req)
        except LookupError:
            respond("<IQ> Sorry, i didn't get that, please try again")
            continue
    else:
        human_req = input("<HOOMAN> ")
    request = tokenizer.texts_to_sequences([human_req])
    # padding the list to have exactly the same shape as embedded #layer input
    request = pad_sequences(request, maxlen=2, dtype="int32", value=0)
    # print(request.shape)
    pred = model.predict(request, batch_size=1, verbose=0)[0]
    pred = list(pred)
    action = pred[0 : len(pred) - 1]
    params = pred[len(pred) - 1]
    response = ""
    # print(action, params, pred)
    if np.argmax(action) == 0:
        response = f"<IQ> ok, {round(params)} minutes, on the timer"
    elif np.argmax(action) == 1:
        response = f"<IQ> it is {time.strftime('%H:%M')}"
    elif np.argmax(action) == 2:
        response = "<IQ> Hello!"

    respond(response)
