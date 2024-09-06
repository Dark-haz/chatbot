from flask import request, jsonify
from services import construct_multishot_prompt, get_completion, generate_embedding
from Repository.ItemRepository import ItemRepository
import boto3
import json




def generate_prompt(input_description, tuples_list):
    # Format the examples
    examples = [
        {
            "input_description": "A bright red apple",
            "tuples_list": [
                (1, "A green apple that is sour"),
                (2, "A bright red juicy apple"),
                (3, "A yellow banana")
            ],
            "output": "ID: 2, Description: 'A bright red juicy apple'"
        },
        {
            "input_description": "A fast black car",
            "tuples_list": [
                (1, "A blue bicycle with a basket"),
                (2, "A black sports car that is very fast"),
                (3, "A white truck for heavy loads")
            ],
            "output": "ID: 2, Description: 'A black sports car that is very fast'"
        }
    ]
    
    # Format the examples into the prompt string
    examples_text = "\n\n".join(
        f"**Example:**\n"
        f"**Input description:** \"{example['input_description']}\"\n"
        f"**List of tuples:**\n"
        f"{' '.join([f'- ({t[0]}, \"{t[1]}\")' for t in example['tuples_list']])}\n"
        f"**Output:**\n"
        f"{example['output']}"
        for example in examples
    )
    
    # Generate the prompt
    prompt = (
        f"You are given an input description and a list of tuples where each tuple contains an ID and a description. "
        f"Your task is to find the tuple whose description matches the input description most closely. "
        f"Return only the ID and the best-matching description, without any additional explanation. Here's how to proceed:\n\n"
        f"{examples_text}\n\n"
        f"**Now, process the following:**\n"
        f"**Input description:** \"{input_description}\"\n"
        f"**List of tuples:**\n"
        f"{' '.join([f'- ({t[0]}, \"{t[1]}\")' for t in tuples_list])}\n\n"
        f"**Return:** The ID and the best-matching description only."
    )
    
    return prompt

REGION_NAME = "us-east-1"
MODEL_NAME = "anthropic.claude-3-5-sonnet-20240620-v1:0"
def get_completion(prompt):
    try:
        bedrock = boto3.client(service_name="bedrock-runtime", region_name='us-east-1')
        body = json.dumps({
            "max_tokens": 100,  
            "messages": [{"role": "user", "content": prompt}],
            "anthropic_version": "bedrock-2023-05-31"
        })

        response = bedrock.invoke_model(body=body, modelId=MODEL_NAME)
        response_body = json.loads(response.get("body").read())
        ss = response_body.get("content")
        
        # Extract ID and description directly
        result = ss[0]['text'].strip()
        return result
    except Exception as e:
        print(f"Error communicating with Claude: {e}")
        raise e
user_input = "I'm looking for a wireless mouse that's comfortable for long use and has a long battery life. My budget is around $30."
metadata = {"gender": "Male", "height": 30.00, "brand": "LG"}
prompt = construct_multishot_prompt(user_input, metadata)
response = get_completion(prompt)

# Generate embedding
embed = generate_embedding(response)

# Fetch KNN results
ss = ItemRepository()
rr = ss.find_KNN(embed, 3)

# Generate prompt for the LLM
prompt2 = generate_prompt(user_input, rr)
response = get_completion(prompt2)

# Print response which should include ID and description
print(response)
