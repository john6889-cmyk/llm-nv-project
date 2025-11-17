import json
import base64
import ast

# from openai import OpenAI
from prompt import meta_prompt
import requests
import re

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_structured_plan(raw_text):
    """Try to recover a JSON-ish block with waypoint info from the model output."""
    if not raw_text:
        return None

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
    candidate = fenced_match.group(1) if fenced_match else raw_text.strip()

    # Fall back to grabbing the first braces pair if extra prose surrounds the JSON.
    if not candidate.startswith("{"):
        brace_block = re.search(r"\{.*\}", candidate, re.DOTALL)
        if brace_block:
            candidate = brace_block.group(0)

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(candidate)
        except (ValueError, SyntaxError):
            return None


def ai_planner():
    # url="http://saltyfish.eecs.umich.edu:8000/v1/chat/completions"
    url = "http://ronaldo.eecs.umich.edu:11400/api/generate"
    headers={"Content-Type":"application/json"}

    # Path to your image
    image_path = "./input.png"
    # Getting the Base64 string
    base64_image = encode_image(image_path)
    
    # Asking for user input
    question = input("Prompt : ")

    # Find a path to the Dining Room.
    
    # response = client.client.beta.assistants.create

    # data = {
    # # "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    # "model":"llava",
    # "messages": [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "text",
    #                 "text":meta_prompt+question,
    #             },
    #             {
    #                 "type": "image_url",
    #                 "image_url":{"url": f"data:image/png;base64,{base64_image}"},
    #             }
    #         ]
    #     }
    # ],
    # "temperature": 0.1  # Lower temperature for more consistent structured output
    # }

    data = {
        "model": "llava",
        "prompt": meta_prompt + question,    # <— /api/generate uses 'prompt'
        "images": [base64_image],                     # <— images as raw base64 list
        "stream": False,                     # <— turn off streaming
        "options": {"temperature": 0.1},
    }
    response=requests.post(url,headers=headers, json=data, timeout=1000)
    response.raise_for_status()
    result=response.json()

    structured_response = extract_structured_plan(result.get("response", ""))
    if not structured_response:
        print("Model response could not be parsed as structured data:\n")
        print(result.get("response", ""))
    return structured_response

    # if "choices" in result and len(result["choices"]) > 0:
    #     content=result["choices"][0]["message"]["content"]

    
    # stripped_content = content.strip()
    # if stripped_content.startswith("```"):
    #     lines = stripped_content.splitlines()
    #     lines = lines[1:]  # drop opening fence
    #     if lines and lines[-1].strip().startswith("```"):
    #         lines = lines[:-1]  # drop closing fence
    #     stripped_content = "\n".join(lines).strip()



    # try:
    #     response_dict=json.loads(stripped_content)
    # except json.JSONDecodeError as exc:
    #     raise ValueError(f"Failed to parse model response as JSON: {exc}\nRaw response: {stripped_content}") from exc

    # return response_dict

if __name__ == "__main__":
    response_dict=ai_planner()
    if not response_dict:
        exit(1)

    goal=response_dict.get("goal")
    reasoning=response_dict.get("reasoning")

    if goal:
        print("Goal waypoints:")
        for idx, waypoint in enumerate(goal, start=1):
            print(f"  {idx}. {waypoint}")
    else:
        print("No goal waypoints found in response.")

    if reasoning:
        print("\nReasoning:")
        print(reasoning)
