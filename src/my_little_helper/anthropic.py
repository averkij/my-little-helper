import json
import json
import anthropic
from anthropic.types.beta.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.beta.messages.batch_create_params import Request
import anthropic


def make_and_process_batch(
    prompts,
    model,
    system="",
    output_path="./batch_anthropic.jsonl",
    temperature=0.2,
    max_tokens=2048,
):
    client = anthropic.Anthropic()
    requests = []
    for prompt in prompts:
        if system:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt["input"]},
            ]
        else:
            messages = [{"role": "user", "content": prompt["input"]}]
        request = Request(
            custom_id=prompt["custom_id"],
            params=MessageCreateParamsNonStreaming(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
                temperature=temperature,
            ),
        )
        requests.append(request)
    with open(output_path, "w", encoding="utf8") as f:
        for task in requests:
            json_string = json.dumps(task, ensure_ascii=False)
            f.write(json_string + "\n")

    batch = client.beta.messages.batches.create(requests=requests)
    print(batch)

    return batch.id


def get_batch_status(batch_name):
    client = anthropic.Anthropic()
    state = client.beta.messages.batches.retrieve(
        batch_name,
    )
    print("Status:", state.processing_status)

    print("*" * 80)
    print("succeeded:", state.request_counts.succeeded)
    print("processing:", state.request_counts.processing)
    print("errored:", state.request_counts.errored)
    print("canceled:", state.request_counts.canceled)
    print("expired:", state.request_counts.expired)

    # if state.status == "failed":
    #     print("Error:", state.errors)
    return state


def get_batch_result(
    batch_name,
    output_path="./batch_res.jsonl",
    batch_content="./batch_content.jsonl",
):
    client = anthropic.Anthropic()

    succeded = 0
    errored = 0
    expired = 0
    succeded_docs = []
    for result in client.beta.messages.batches.results(
        batch_name,
    ):
        match result.result.type:
            case "succeeded":
                succeded += 1
                succeded_docs.append(result)
            case "errored":
                errored += 1
                if result.result.error.type == "invalid_request":
                    print(f"Validation error {result.custom_id}")
                else:
                    print(f"Server error {result.custom_id}")
            case "expired":
                expired += 1
                print(f"Request expired {result.custom_id}")

    print("completed:", succeded)
    print("errored:", errored)
    print("expired:", expired)
    
    with open(output_path, "w", encoding="utf8") as output_file:
        for doc in succeded_docs:
            output_file.write(doc.model_dump_json() + "\n")

    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    res_docs = []
    for doc in succeded_docs:
        res_docs.append(doc.result.message.content)
        usage = doc.result.message.usage
        input_tokens += usage.input_tokens
        output_tokens += usage.output_tokens
        total_tokens += usage.input_tokens
        total_tokens += usage.output_tokens

    with open(batch_content, "w", encoding="utf8") as content_file:
        for doc in res_docs:
            content_file.write(json.dumps(doc[0].text, ensure_ascii=False) + "\n")

    print("*" * 80)
    print("input_tokens:", input_tokens)
    print("output_tokens:", output_tokens)
    print("total_tokens:", total_tokens)

    return
