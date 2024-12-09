import json
import subprocess
import os
import json
from openai import OpenAI
from datetime import datetime


# os.environ["OPENAI_API_KEY"] = ""


def make_tasks(prompts, model, output_path="./tasks.jsonl"):
    tasks = []
    for prompt in prompts:
        task = {
            "model": model,
            "messages": [{"role": "user", "content": prompt["input"]}],
            "metadata": prompt["meta"],
        }
        tasks.append(task)
    with open(output_path, "w", encoding="utf8") as f:
        for task in tasks:
            json_string = json.dumps(task, ensure_ascii=False)
            f.write(json_string + "\n")


def do_tasks_parallel(tasks_path, result_path="./result.jsonl"):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_processor_path = os.path.join(current_dir, "api_request_parallel_processor.py")
    command = [
        "python",
        api_processor_path,
        "--requests_filepath",
        tasks_path,
        "--save_filepath",
        result_path,
        "--request_url",
        "https://api.openai.com/v1/chat/completions",
        "--max_requests_per_minute",
        "500",
        "--max_tokens_per_minute",
        "200000",
        "--token_encoding_name",
        "o200k_base",
        "--max_attempts",
        "5",
        "--logging_level",
        "10",
    ]
    print("Running the following command: ", " ".join(command))
    subprocess.run(command, check=True)


def make_batch(
    prompts,
    model,
    system="",
    output_path="./batch_openai.jsonl",
    schema={},
    temperature=0.2,
):
    tasks = []
    for prompt in prompts:
        if system:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt["input"]},
            ]
        else:
            messages = [{"role": "user", "content": prompt["input"]}]
        task = {
            "custom_id": prompt["custom_id"],
            "method": "POST",
            "body": {"model": model, "messages": messages, "temperature": temperature},
            "url": "/v1/chat/completions",
        }
        if schema:
            task["body"]["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema["name"],
                    "schema": schema,
                    "strict": True,
                },
            }
        tasks.append(task)
    with open(output_path, "w", encoding="utf8") as f:
        for task in tasks:
            json_string = json.dumps(task, ensure_ascii=False)
            f.write(json_string + "\n")


def generate_json_schema(json_obj):
    """
    Generate a JSON schema from a given JSON object.

    :param json_obj: The JSON object for which the schema is to be generated.
    :return: A dictionary representing the JSON schema.
    """

    def infer_type(value):
        """Infer the JSON type of a value."""
        if isinstance(value, dict):
            return "object"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, bool):
            return "boolean"
        elif value is None:
            return "null"
        else:
            raise ValueError(f"Unsupported data type: {type(value)}")

    def generate_schema(obj):
        """Recursively generate schema for a JSON object."""
        schema = {"type": infer_type(obj)}

        if schema["type"] == "object":
            schema["properties"] = {}
            for key, value in obj.items():
                schema["properties"][key] = generate_schema(value)
        elif schema["type"] == "array":
            if obj:
                schema["items"] = generate_schema(obj[0])
            else:
                schema["items"] = {}  # No items to infer
        return schema

    return {
        "name": "my_schema",
        "type": "object",
        "properties": generate_schema(json_obj)["properties"],
        "required": list(json_obj.keys()),
        "additionalProperties": False,
    }


def proces_batch(fpath, batch_desc=""):
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )
    batch = client.files.create(
        file=open(fpath, "rb"),
        purpose="batch",
    )
    print("batch id:", batch.id)
    with open("./log.txt", "a") as log:
        log.write(f"[{datetime.now()}] batch id: {batch.id}\n")

    task = client.batches.create(
        input_file_id=batch.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": batch_desc},
    )
    print("batch name:", task.id)
    with open("./log.txt", "a") as log:
        log.write(f"[{datetime.now()}] batch name: {task.id}\n")

    return task.id

def get_batch_status(task_id):
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )
    state = client.batches.retrieve(task_id)
    print("Status:", state.status)
    
    print('*'*80)
    print("completed:", state.request_counts.completed)
    print("failed:", state.request_counts.failed)
    print("total:", state.request_counts.total)

    if state.status == "failed":
        print("Error:", state.errors)
    return client.batches.retrieve(task_id)


def get_batch_result(
    task_id,
    output_path="./batch_res.jsonl",
    batch_content="./batch_content.jsonl",
    output_error_path="./batch_res_error.jsonl",
):
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )
    state = client.batches.retrieve(task_id)
    print("Status:", state.status)

    if state.request_counts.completed > 0:
        output_file_id = state.output_file_id
        content = client.files.content(output_file_id)
        content_str = content.read()
        with open(output_path, "wb") as json_file:
            json_file.write(content_str)
    elif state.request_counts.failed > 0:
        output_file_id = state.error_file_id
        content = client.files.content(output_file_id)
        content_str = content.read()
        with open(output_error_path, "wb") as json_file:
            json_file.write(content_str)

    print("completed:", state.request_counts.completed)
    print("failed:", state.request_counts.failed)

    res_docs = []
    with open(output_path, "r", encoding="utf-8") as fin:
        res_json = fin.readlines()
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        for item in res_json:
            doc = json.loads(item)
            res_docs.append(doc["response"]["body"]["choices"][0]["message"]["content"])
            usage = doc["response"]["body"]["usage"]
            input_tokens += usage["prompt_tokens"]
            output_tokens += usage["completion_tokens"]
            total_tokens += usage["total_tokens"]
        
    with open(batch_content, "w", encoding="utf8") as content_file:
        for doc in res_docs:
            content_file.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print("*" * 80)
    print("input_tokens:", input_tokens)
    print("output_tokens:", output_tokens)
    print("total_tokens:", total_tokens)

    return