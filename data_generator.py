# import random


# data_csv = open('training_data.csv', 'a')
# data_csv.write('request,action,params\n')
# training_data = []
# for i in range(1, 61):
#     training_data.append(f"{i} mins,1,{i}\n")
# for i in range(1, 61):
#     training_data.append(f"{i} minutes,1,{i}\n")
# for i in range(1, 61*2):
#     training_data.append('hello,3,0\n')
#     training_data.append('hi,3,0\n')
#     training_data.append('what time,2,0\n')
#     training_data.append('what time is it,2,0\n')
# # training_data.append('hello,3,0\n')
# # training_data.append('hi,3,0\n')
# # training_data.append('what time,2,0')
# random.shuffle(training_data)
# data_csv.writelines([*(training_data)])
# #1 mins,1,1
# # 2 mins,1,2
# # 3 mins,1,3
# # 4 mins,1,4
# # 5 mins,1,5
# # what time,2,0
# # hello,3,0

import json
import random
from itertools import cycle, islice
from utils.errors import UnevenDataErr

total_lines = 500

intents = ["age", "greeting", "name", "openexe", "name"]
total_intents = []
# for intent in intents:
#     responses = []
#     patterns = []
#     while True:
#         a = input(f"{intent} -- Responses: ")
#         if a == "**done**":
#             break
#         else:
#             responses.append(a)
#     while True:
#         a = input(f"{intent} -- Patterns: ")
#         if a == "**done**":
#             break
#         else:
#             patterns.append(a)
#     total_intents.append({"tag": intent, "patterns": patterns, "responses": responses})

total_intents = json.load(open("training_data/training_data.v4.iq.json"))["intents"]
for i, intent in enumerate(total_intents):
    if intent["tag"] == "starttimer":
        intent_patterns = intent["patterns"]
        times_to_copy = round(total_lines / 4)
        print(times_to_copy)
        training_data = []
        for i2 in range(1, times_to_copy):
            training_data.append(f"{i2} mins")
            training_data.append(f"{i2} secs")
        for i2 in range(1, times_to_copy):
            training_data.append(f"{i2} minutes")
            training_data.append(f"{i2} seconds")
        print(training_data, i)
        random.shuffle(training_data)
        total_intents[i]["patterns"] = training_data
        continue
    intent_patterns = intent["patterns"]
    # times_to_copy = round(total_lines / len(intent_patterns) - 1)
    # intent_patterns.extend(intent_patterns * times_to_copy)
    # print(intent_patterns)
    total_intents[i]["patterns"] = list(islice(cycle(intent_patterns), total_lines))
print(len(total_intents[0]["patterns"]))
if (
    not len(total_intents[0]["patterns"])
    == len(total_intents[1]["patterns"])
    == len(total_intents[2]["patterns"])
    == len(total_intents[3]["patterns"])
    == len(total_intents[4]["patterns"])
):
    print(
        len(total_intents[0]["patterns"]),
        len(total_intents[1]["patterns"]),
        len(total_intents[2]["patterns"]),
        len(total_intents[3]["patterns"]),
        len(total_intents[4]["patterns"]),
    )
    raise UnevenDataErr
json.dump(
     {"intents": total_intents}, open("training_data/training_data.v4.iq.json", "w")
)
