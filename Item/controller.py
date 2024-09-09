from flask import request, jsonify
import json
from Item.services import construct_multishot_prompt, get_completion, generate_embedding,generate_prompt
from Item.Repository.ItemRepository import ItemRepository
def execution():
    user_input = request.json.get("user_input")
    metadata = request.json.get("metadata")

        # Construct the prompt using the provided input and metadata
    prompt = construct_multishot_prompt(user_input, metadata)
    response = get_completion(prompt)
    embed= generate_embedding(response[0])
    ss=ItemRepository()
    rr=ss.find_KNN(embed,3)
    prompt2= generate_prompt(response,rr)
    result=get_completion(prompt2)
    print(response)
    return result
    
    




