import json
import boto3

def get_bedrock_client():
    """Initialize and return the Bedrock client."""
    return boto3.client(service_name="bedrock-runtime", region_name='us-east-1')

def generate_embedding(prompt_data):
    modelId = "amazon.titan-embed-text-v2:0"
    accept = "application/json"
    contentType = "application/json"

    # Define the input data
    sample_model_input = {
        "inputText": prompt_data,
        "dimensions": 256,
        "normalize": True
    }

    # Convert input data to JSON
    body = json.dumps(sample_model_input)

    # Get the Bedrock client
    bedrock_client = get_bedrock_client()

    # Make the API request
    response = bedrock_client.invoke_model(
        body=body,
        modelId=modelId,
        accept=accept,
        contentType=contentType
    )

    # Process the response
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get("embedding")
    
    # Print the embedding details
    print(f"The embedding vector has {len(embedding)} values\n{embedding[0:3]+['...']+embedding[-3:]}")

# Example usage
prompt_data = "Amazon Bedrock supports foundation models from industry-leading providers such as AI21 Labs, Anthropic, Stability AI, and Amazon. Choose the model that is best suited to achieving your unique goals."
generate_embedding(prompt_data)
