import pandas as pd
import tqdm
import torch
import gc
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.amp import autocast

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def setup_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2-1.5B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto"
    )

    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()

    model.eval()

    return tokenizer, model

def clear_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def generate_rectified_msg(diff, human_commit, llm_inference, tokenizer, model):
        clear_memory()

        messages = [
            {
                "role": "system",
                "content": "You are a fair and impartial model. Analyze the given diff and two commit messages to determine the most accurate one. IF AND ONLY IF both messages have issues, create a better commit message that accurately reflects the changes. Output only the final commit message without quotes or special characters."
            },
            {
                "role": "user",
                "content": f"""Given this diff and two commit messages, choose the better one or create an improved version IF AND ONLY IF ITS REQUIRED.
                    Diff: {diff}
                    Human: {human_commit}
                    LLM: {llm_inference}

                    Output the best commit message:"""
            }
        ]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            max_length=2048,
            truncation=True
        )

        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad(), autocast("cuda"):
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=False
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        ).replace("<|im_end|>", "").strip()

        del inputs, outputs
        clear_memory()

        return response


def process_data(data, tokenizer, model, freq=50):
    ans = []

    for i in tqdm.tqdm(range(len(data))):
        result = generate_rectified_msg(
            data.loc[i]["Diff"],
            data.loc[i]["Message"],
            data.loc[i]["LLM Inference"],
            tokenizer,
            model
        )
        ans.append(result)

        if (i + 1) % freq == 0: #save every freq its
            temp_data = data.iloc[:i+1].copy()
            temp_data["rectified message"] = ans
            temp_data.to_csv("commits_with_rectified_msgs_temp.csv", escapechar='\\', index=False)
            clear_memory()

    return ans

clear_memory()
tokenizer, model = setup_model_and_tokenizer()
data = pd.read_csv("data_with_llm_inferences.csv", escapechar='\\')
ans = process_data(data, tokenizer, model)

data["rectified message"] = ans
data.to_csv("commits_with_rectified_msgs.csv", escapechar='\\', index=False)