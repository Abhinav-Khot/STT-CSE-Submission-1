from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import numpy as np
import tqdm
import gc

tokenizer = AutoTokenizer.from_pretrained("mamiksik/CommitPredictorT5")
model = AutoModelForSeq2SeqLM.from_pretrained("mamiksik/CommitPredictorT5").cuda()

'''
    We will first generate the LLM inferences and then proceed with the rectification process
'''

def generate_commit_message(code_snippet, max_length=100):
    '''
        Provides LLM inference given a code snippet. Max length of the LLM response is 100 by defualt.
    '''
    inputs = tokenizer.encode(code_snippet, return_tensors="pt").cuda()
    outputs = model.generate(inputs, max_length=max_length, num_beams=4, early_stopping=True)
    ret = tokenizer.decode(outputs[0], skip_special_tokens=True)
    del inputs
    del outputs
    gc.collect()
    return ret

data = pd.read_csv("commit_files_diff.csv", escapechar='\\')
llm_inferences = []

for row in tqdm.tqdm(data.itertuples()):
    llm_inferences.append(generate_commit_message(row.Diff))

data['LLM Inference'] = llm_inferences
data.to_csv('data_with_llm_inferences.csv', escapechar='\\')