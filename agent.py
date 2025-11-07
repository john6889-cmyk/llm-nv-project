import json
import base64

# from openai import OpenAI
from prompt import meta_prompt
import requests
import re

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def ai_planner():
    url="http://saltyfish.eecs.umich.edu:8000/v1/chat/completions"
    headers={"Content-Type":"application/json"}

    # Path to your image
    image_path = "./input.png"
    # Getting the Base64 string
    base64_image = encode_image(image_path)
    
    # Asking for user input
    question = input("Prompt : ")

    # Find a path to the Dining Room.
    
    # response = client.client.beta.assistants.create

    data = {
    "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text":meta_prompt+question,
                },
                {
                    "type": "image_url",
                    "image_url":{"url": f"data:image/png;base64,{base64_image}"},
                }
            ]
        }
    ],
    "temperature": 0.1  # Lower temperature for more consistent structured output
    }

    response=requests.post(url,headers=headers, json=data, timeout=180)
    response.raise_for_status()
    result=response.json()
    # print(result)

    if "choices" in result and len(result["choices"]) > 0:
        content=result["choices"][0]["message"]["content"]

    
    stripped_content = content.strip()
    if stripped_content.startswith("```"):
        lines = stripped_content.splitlines()
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]  # drop closing fence
        stripped_content = "\n".join(lines).strip()



    try:
        response_dict=json.loads(stripped_content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse model response as JSON: {exc}\nRaw response: {stripped_content}") from exc

    return response_dict

if __name__ == "__main__":
    response_dict=ai_planner()
    print(response_dict)
