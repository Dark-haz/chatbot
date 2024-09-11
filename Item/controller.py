from flask import request, jsonify
import json
from Item.services import process_user_input_prompt, invoke_bedrock_claude, invoke_bedrock_titan, recommend_product_prompt
from Item.Repository.ItemRepository import ItemRepository
import xml.etree.ElementTree as ET
import xmltodict
import json
from Item.input_schema import validate_user_input_data
def execution():
    # Middleware for validation
    validation_error = validate_user_input_data()
    if validation_error:
        return validation_error

    user_input = request.json.get("user_input")
    metadata = request.json.get("metadata")

    user_input_processing_prompt = process_user_input_prompt(user_input, metadata)
    
    structured_user_query = invoke_bedrock_claude(user_input_processing_prompt, 100).split('\n')[0]

    user_input_embed = invoke_bedrock_titan(structured_user_query)
    
    item_repository = ItemRepository()
    similar_items = item_repository.find_KNN(user_input_embed, 10)

    product_selection_prompt = recommend_product_prompt(structured_user_query, similar_items)
    result = invoke_bedrock_claude(product_selection_prompt, 1000)

    try:
        root = ET.fromstring(result)
    except:
        return result

    data_dict = xmltodict.parse(result)

    json_data = json.dumps(data_dict, indent=4)

    return json_data
