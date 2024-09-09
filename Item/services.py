import boto3
import json

# USER SIDE

# > step 1 : Claude user input processing 
def construct_multishot_prompt(user_input, metadata):
    # Define the few-shot examples
    few_shot_examples = """
    Example 1:
    Unstructured Input: "I'm looking for a wireless mouse that's comfortable for long use and has a long battery life. My budget is around $30."
    Metadata: {"gender": "Male", "previous_brand": "LG"}
    Descriptive Output:
    Wireless mouse that is comfortable for long-term use and has an extended battery life. With a budget of around $30, brand LG.

    Example 2:
    Unstructured Input: "I need a portable Bluetooth speaker, something that sounds great and can last for a few hours on a charge."
    Metadata: {"gender": "Female", "previous_brand": "Sony"}
    Descriptive Output:
    Portable Bluetooth speaker that delivers excellent sound quality and has a long battery life. Brand Sony.

    Example 3:
    Unstructured Input: "I need a laptop."
    Metadata: {"gender": "Female", "previous_brand": "Sony"}
    Descriptive Output:
    Laptop from brand Sony.

    example 4 :
     Unstructured Input: "I need a green hat."
    Metadata: {"gender": "Female", "previous_brand": "Nike"}
    Descriptive Output:
   
    """
    
    # Format the user input and metadata
    user_example = f"""
    Unstructured Input: "{user_input}"
    Metadata: {metadata}
    Descriptive Output:
    """
    
    # Combine everything into the final prompt
    final_prompt = few_shot_examples + "\n" + user_example
    
    return final_prompt

def get_completion(prompt):
    try:
        # bedrock = boto3.client(service_name="bedrock-runtime", region_name=REGION_NAME)
        bedrock = boto3.client(service_name="bedrock-runtime", region_name='us-east-1')
        body = json.dumps({
            "max_tokens": 100,  
            "messages": [{"role": "user", "content": prompt}],
            "anthropic_version": "bedrock-2023-05-31"
        })

        response = bedrock.invoke_model(body=body, modelId="anthropic.claude-3-5-sonnet-20240620-v1:0")
        response_body = json.loads(response.get("body").read())
        ss= response_body.get("content")
        sr = ss[0]['text'].split('\n')
        return sr
    except Exception as e:
        print(f"Error communicating with Claude: {e}")
        raise e
    
# > step 2 : TITAN embedd user input 

def generate_embedding(prompt_data):
    modelId = "amazon.titan-embed-text-v2:0"
    accept = "application/json"
    contentType = "application/json"

    
    sample_model_input = {
        "inputText": prompt_data,
        "dimensions": 256,
        "normalize": True
    }

    
    body = json.dumps(sample_model_input)

    
    bedrock_client = boto3.client(service_name="bedrock-runtime", region_name='us-east-1')

    
    response = bedrock_client.invoke_model(
        body=body,
        modelId=modelId,
        accept=accept,
        contentType=contentType
    )

   
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get("embedding")
    return embedding
    

# > step 2 : find by KNN using processed user input vector (inside controller)

# > step 3 : Claude product selection
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



def main():
    user_input = "  I'm looking for a wireless mouse that's comfortable for long use and has a long battery life. and a pents that's comfy to wear at work. very fast car yes yes"
    metadata = {"gender": "Male", "height": 30.00,"brand":"LG"}
    prompt = construct_multishot_prompt(user_input, metadata)
    response = get_completion(prompt)
    print(response)    
    # embed= generate_embedding(response[0])
    # print(embed)

if __name__ == "__main__":
    main()
