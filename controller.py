from flask import request, jsonify
from services import construct_multishot_prompt, get_completion, generate_embedding
from Repository.ItemRepository import ItemRepository

def execution():
    user_input = request.json.get("user_input")
    metadata = request.json.get("metadata")

    # Construct the prompt using the provided input and metadata
    prompt = construct_multishot_prompt(user_input, metadata)
    response = get_completion(prompt)

    embed= generate_embedding(response[0])
    ss=ItemRepository()
    rr=ss.find_KNN(embed,3)
    print(rr)



