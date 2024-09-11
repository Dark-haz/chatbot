from marshmallow import Schema, fields, ValidationError
from flask import request, jsonify
class UserInputSchema(Schema):
    user_input = fields.Str(required=True)
    metadata = fields.Dict(keys=fields.Str(), values=fields.Raw(), required=True)
def validate_user_input_data():
    schema = UserInputSchema()
    data = request.get_json()
    
    try:
        schema.load(data)
    except ValidationError as err:
        return jsonify({"error": err.messages}), 400
    
    return None  # No error
