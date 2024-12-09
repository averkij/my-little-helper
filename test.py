# %%
import importlib
import json
import os

from src import my_little_helper as m
from src.my_little_helper import openai, anthropic

importlib.reload(openai)
importlib.reload(anthropic)


fpath = "./ru_alpaca_seed_tasks.jsonl"

docs = [json.loads(line) for line in open(fpath, "r", encoding="utf-8")]

prompts = []
for doc in docs[:125]:
    prompt = "Translate values of given JSON object to Tatar. Use more native Tatar vocabulary and less Russian vocab. Return only translated JSON object and nothing else. JSON:\n\n%%TEXT%%"

    obj = {
        "inst": doc["orig_instruction"],
        "input": doc["orig_instances"][0]["input"],
        "output": doc["orig_instances"][0]["output"],
    }

    prompt = prompt.replace("%%TEXT%%", json.dumps(obj, indent=4, ensure_ascii=False))
    prompts.append(
        {
            "input": prompt,
            "meta": {"row_id": doc["id"], "type": "orig_instances"},
            "custom_id": doc["id"],
        }
    )

#%%
#OPENAI

obj = {
    "inst": "some string1",
    "input": "some string2",
    "output": "some string3",
}
schema = m.openai.generate_json_schema(obj)
schema

# %%

m.openai.make_batch(prompts, model="gpt-4o", schema=schema)

# %%
batch_name = m.openai.proces_batch(
    fpath="./batch_openai.jsonl", batch_desc="Tatar translation"
)
# %%
m.openai.get_batch_status("batch_675563d511b881919cd6e520ff20bde2")

# %%
m.openai.get_batch_result(
    task_id="batch_675563d511b881919cd6e520ff20bde2",
    output_path="./batch_res_openai.jsonl",
    batch_content="./batch_content_openai.jsonl",
)

# %%
# ANTHROPIC

#claude-3-5-sonnet-20241022
#claude-3-5-haiku-20241022

batch_name = anthropic.make_and_process_batch(prompts, model="claude-3-5-sonnet-20241022")

# %%
anthropic.get_batch_status(batch_name=batch_name)

# %%
anthropic.get_batch_result(batch_name=batch_name,
    output_path="./batch_res_anthropic.jsonl",
    batch_content="./batch_content_anthropic.jsonl")

# %%
