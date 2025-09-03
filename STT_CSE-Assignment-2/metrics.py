import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModel.from_pretrained("microsoft/codebert-base")

model.eval()
model.to('cuda')

def generate_embedding(message):
    message = str(message).strip()
    marked_text = "<s>" + message + "</s>"

    tokenized_text = tokenizer.tokenize(marked_text, truncation=True, max_length=tokenizer.model_max_length)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens]).to('cuda')

    with torch.no_grad():
        outputs = model(tokens_tensor)
        last_hidden_states = outputs[0]
        sentence_embedding = last_hidden_states[0, 0, :]

    result = sentence_embedding.cpu().numpy()

    del tokens_tensor, outputs, last_hidden_states, sentence_embedding
    torch.cuda.empty_cache()

    return result

data = pd.read_csv("commits_with_rectified_msgs_(prefixes removed).csv", escapechar='\\')

human_embeddings = np.array([generate_embedding(msg) for msg in data["Message"].tolist()])
llm_embeddings = np.array([generate_embedding(msg) for msg in data["LLM Inference"].tolist()])
rectified_embeddings = np.array([generate_embedding(msg) for msg in data["rectified message"].tolist()])
diff_embeddings = np.array([generate_embedding(msg) for msg in data["Diff"].tolist()])

def rowwise_cosine_similarity(a, b):
    sims = []
    for i in range(len(a)):
        sims.append(cosine_similarity(a[i].reshape(1, -1), b[i].reshape(1, -1))[0, 0])
    return np.mean(sims)

diff_human_similarity = rowwise_cosine_similarity(diff_embeddings, human_embeddings)
diff_llm_similarity = rowwise_cosine_similarity(diff_embeddings, llm_embeddings)
diff_rectified_similarity = rowwise_cosine_similarity(diff_embeddings, rectified_embeddings)

print(f"Hit rate for human commit messages: {diff_human_similarity}")
print(f"Hit rate for llm commit messages: {diff_llm_similarity}")
print(f"Hit rate for rectified commit messages: {diff_rectified_similarity}")