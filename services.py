import boto3
import json
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



REGION_NAME = "us-east-1"
MODEL_NAME = "anthropic.claude-3-5-sonnet-20240620-v1:0"
def get_completion(prompt):
    try:
        # bedrock = boto3.client(service_name="bedrock-runtime", region_name=REGION_NAME)
        bedrock = boto3.client(service_name="bedrock-runtime", region_name='us-east-1')
        body = json.dumps({
            "max_tokens": 100,  
            "messages": [{"role": "user", "content": prompt}],
            "anthropic_version": "bedrock-2023-05-31"
        })

        response = bedrock.invoke_model(body=body, modelId=MODEL_NAME)
        response_body = json.loads(response.get("body").read())
        ss= response_body.get("content")
        sr = ss[0]['text'].split('\n')
        return sr
    except Exception as e:
        print(f"Error communicating with Claude: {e}")
        raise e


import boto3
import json



def get_bedrock_client():
    """Initialize and return the Bedrock client."""
    return boto3.client(service_name="bedrock-runtime", region_name='us-east-1')

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

    
    bedrock_client = get_bedrock_client()

    
    response = bedrock_client.invoke_model(
        body=body,
        modelId=modelId,
        accept=accept,
        contentType=contentType
    )

   
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get("embedding")
    return embedding
    




def main():
    user_input = "laptop."
    metadata = {"gender": "Male", "height": 30.00}
    prompt = construct_multishot_prompt(user_input, metadata)
    response = get_completion(prompt)
    print(response)    
    embed= generate_embedding(response[0])
    print(embed)

if __name__ == "__main__":
    main()
