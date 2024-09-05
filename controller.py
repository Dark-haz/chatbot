from flask import request, jsonify
from services import construct_claude_prompt_for_description, get_completion, generate_embedding
def execution():
    user_input = request.json.get("user_input")
    metadata = request.json.get("metadata")

        # Construct the prompt using the provided input and metadata
    prompt = construct_claude_prompt_for_description(user_input, metadata)
    response = get_completion(prompt)
    embed= generate_embedding(response[0])
    return jsonify({"response": response,
                    "embedding": embed})