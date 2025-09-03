#the rectification LLM after choosing the best commit message, outputs Human: or LLM: as a prefix or no prefx at all. In case the prefix is present we need to remove it.
import pandas as pd

def clean_message(message):
    if message[0] == '"':
        message = message[1:-1]
    if message.startswith("Human: "):
        message = message[len("Human: "):]
    elif message.startswith("LLM: "):
        message = message[len("LLM: "):]
    return message

data = pd.read_csv("commits_with_rectified_msgs.csv", escapechar='\\')
data['rectified message'] = data['rectified message'].apply(clean_message)
data.to_csv("commits_with_rectified_msgs_(prefixes removed).csv", escapechar='\\')