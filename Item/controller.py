from flask import request, jsonify
import json
from Item.services import process_user_input_prompt, invoke_bedrock_claude, invoke_bedrock_titan,recommend_product_prompt , invoke_bedrock_claude
from Item.Repository.ItemRepository import ItemRepository
def execution():
    user_input = request.json.get("user_input")
    metadata = request.json.get("metadata")

    user_input_processing_prompt = process_user_input_prompt(user_input, metadata)
    
    structured_user_input = invoke_bedrock_claude(user_input_processing_prompt,100).split('\n')[0]

    user_input_embed = invoke_bedrock_titan(structured_user_input)
    
    item_repository=ItemRepository()
    similar_items = item_repository.find_KNN(user_input_embed,10)

    product_selection_prompt= recommend_product_prompt(structured_user_input,similar_items)
    result=invoke_bedrock_claude(product_selection_prompt , 500)
    return result
    
    
