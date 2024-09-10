import boto3
import json
from config.static import *
# USER SIDE

# > step 1 : Claude user input processing 
def process_user_input_prompt(user_input, metadata):
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

    Example 4:
    Unstructured Input: "I need a green hat."
    Metadata: {"gender": "Female", "previous_brand": "Nike"}
    Descriptive Output:
    Green hat from brand Nike.
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

def invoke_bedrock_claude(prompt, max_tokens): 
    try: 
        bedrock = boto3.client(service_name="bedrock-runtime", region_name='us-east-1') 
        body = json.dumps({ 
            "max_tokens": max_tokens,   
            "messages": [{"role": "user", "content": prompt}], 
            "anthropic_version": "bedrock-2023-05-31" 
        }) 
 
        response = bedrock.invoke_model(body=body, modelId="anthropic.claude-3-5-sonnet-20240620-v1:0") 
        response_body = json.loads(response.get("body").read()) 
        response = response_body.get("content") 
        response_dict = response[0]

        model_text_output = response_dict['text'] 
        return model_text_output 
    except Exception as e: 
        print(f"Error communicating with Claude: {e}") 
        raise e
    
# > step 2 : TITAN embedd user input 

def invoke_bedrock_titan(prompt_data):
    modelId = "amazon.titan-embed-text-v2:0"
    accept = "application/json"
    contentType = "application/json"

    
    sample_model_input = {
        "inputText": prompt_data,
        "dimensions": VECTOR_DIMENSION,
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
def recommend_product_prompt(user_query , available_items) :
    persona = """
    You are a highly skilled and experienced professional salesman with extensive knowledge of various products. 
    Your job is to provide personalized product recommendations based on the customer's needs. 
    You understand how to analyze product features, match them to user requirements, and persuade customers with thoughtful and helpful suggestions. 
    You always remain courteous, helpful, and detail-oriented.
    """


    input_format = """
    Example Input Format:
    {
        "user_query": "user query",
        "item_tuple_list": [
            (ItemId1, "Item description 1"),
            (ItemId2, "Item description 2"),
            (ItemId3, "Item description 3")
        ]
    }
    """

    xml_input_format = """
    <user_input>
        <user_query>user query</user_query>
        <item_tuple_list>
            <item>
                <id>ItemId1</id>
                <description>Item description 1</description>
            </item>
            <item>
                <id>ItemId2</id>
                <description>Item description 2</description>
            </item>
            <item>
                <id>ItemId3</id>
                <description>Item description 3</description>
            </item>
        </item_tuple_list>
    </user_input>
"""

    context = f"""
    You will receive a user input in the following format:

    1. **user_query**: A string describing what the user is looking for. This query outlines the user's needs, preferences, or requirements.

    2. **item_tuple_list**: A list of tuples where each tuple contains:
    - **Item ID**: A unique identifier for the item.
    - **Item Description**: A string that describes the item, including its features, characteristics, and any other relevant details.

    {xml_input_format}
    """



    bulleted_output_format = """
    Example Output Format:

    - **Item ID**: [ItemId]
    - **Item Name**: [Derived Item Name]
    
    I recommend this [Derived Item Name] because it [briefly explain its main features and benefits]. This item closely matches your needs and offers [mention how it aligns with the user query].
    """

    xmp_output_format = """
    <recommendations>
        <item>
            <id>[ItemId]</id>
            <name>[Derived Item Name]</name>
            <description>I recommend this [Derived Item Name] because it [briefly explain its main features and benefits]. This item closely matches your needs and offers [mention how it aligns with the user query].</description>
        </item>
    </recommendations>
    """

    tone = """
    The tone of the response should be friendly and enthusiastic, reflecting professionalism and attentiveness.
    """

    task = f"""
    Your task is to analyze the user query and compare it with the descriptions in the item tuple list. 
    Select the most relevant items that align with the user's needs based on their query. 
    For each selected item, generate a recommendation response in the following format:

    - Provide the **Item ID**.
    - Derive and present the **Item Name** based on the description.
    - As a professional salesman, give a persuasive explanation of why the user would want this item, using the description to highlight its features and benefits. Keep the explanation concise and ensure it does not exceed two lines.

    {tone}

    Present the response using XML, with one <item> element per recommended item inside the <recommendations> parent element.

    {xmp_output_format}
    """

    bulleted_examplars = """
    Example 1:
    Input:
    {
        "user_query": "I need a gadget that helps with fitness tracking and is water-resistant.",
        "item_tuple_list": [
            (1, "A sleek fitness tracker with heart rate monitoring and water resistance"),
            (2, "A stylish smartwatch with a large screen and multiple apps"),
            (3, "A basic pedometer that counts steps but is not water-resistant")
        ]
    }

    Output:
    - **Item ID**: 1
    - **Item Name**: Fitness Tracker
    
    I recommend this Fitness Tracker because it offers comprehensive heart rate monitoring and is water-resistant, making it perfect for tracking your fitness activities even in wet conditions. Its sleek design also adds to its appeal.

    Example 2:
    Input:
    {
        "user_query": "I'm looking for a durable backpack for hiking with plenty of storage space.",
        "item_tuple_list": [
            (1, "A rugged hiking backpack with multiple compartments and waterproof material"),
            (2, "A casual daypack with a simple design and limited storage"),
            (3, "A stylish messenger bag suitable for urban use but not designed for hiking")
        ]
    }

    Output:
    - **Item ID**: 1
    - **Item Name**: Rugged Hiking Backpack
    
    I recommend this Rugged Hiking Backpack because it features multiple compartments and is made of waterproof material, providing both durability and ample storage for all your hiking needs. Its rugged design ensures it can withstand tough conditions.
    """

    xml_examplars = """
    <examplars>
        <example>
            <user_input>
                <user_query>I need a gadget that helps with fitness tracking and is water-resistant.</user_query>
                <item_tuple_list>
                    <item>
                        <id>1</id>
                        <description>A sleek fitness tracker with heart rate monitoring and water resistance</description>
                    </item>
                    <item>
                        <id>2</id>
                        <description>A stylish smartwatch with a large screen and multiple apps</description>
                    </item>
                    <item>
                        <id>3</id>
                        <description>A basic pedometer that counts steps but is not water-resistant</description>
                    </item>
                </item_tuple_list>
            </user_input>
            
            <output>
                <recommendation>
                    <id>1</id>
                    <name>Fitness Tracker</name>
                    <description>I recommend this Fitness Tracker because it offers comprehensive heart rate monitoring and is water-resistant, making it perfect for tracking your fitness activities even in wet conditions. Its sleek design also adds to its appeal.</description>
                </recommendation>
            </output>
        </example>

        <example>
            <user_input>
                <user_query>I'm looking for a durable backpack for hiking with plenty of storage space.</user_query>
                <item_tuple_list>
                    <item>
                        <id>1</id>
                        <description>A rugged hiking backpack with multiple compartments and waterproof material</description>
                    </item>
                    <item>
                        <id>2</id>
                        <description>A casual daypack with a simple design and limited storage</description>
                    </item>
                    <item>
                        <id>3</id>
                        <description>A stylish messenger bag suitable for urban use but not designed for hiking</description>
                    </item>
                </item_tuple_list>
            </user_input>
            
            <output>
                <recommendation>
                    <id>1</id>
                    <name>Rugged Hiking Backpack</name>
                    <description>I recommend this Rugged Hiking Backpack because it features multiple compartments and is made of waterproof material, providing both durability and ample storage for all your hiking needs. Its rugged design ensures it can withstand tough conditions.</description>
                </recommendation>
            </output>
        </example>
        
    </examplars>
    """


    prompt = persona + context + task + xml_examplars  

    user_input = f"""
    Input:
    {{
        "user_query": {user_query},
        "item_tuple_list": {available_items}
    }}
    """

    generate_items_xml = lambda item_tuple_list: "".join(
    f"""
    <item>
        <id>{item[0]}</id>
        <description>{item[1]}</description>
    </item>""" for item in item_tuple_list
)

    xml_user_input = f"""
    <user_input>
        <user_query>{user_query}</user_query>
        <item_tuple_list>{generate_items_xml(available_items)}
        </item_tuple_list>
    </user_input>
    """

    # Appending user input to the prompt
    full_prompt = prompt + xml_user_input
    return full_prompt



def main():
    # user_input = "  I'm looking for a wireless mouse that's comfortable for long use and has a long battery life. and a pents that's comfy to wear at work. very fast car yes yes"
    # metadata = {"gender": "Male", "height": 30.00,"brand":"LG"}
    # prompt = process_user_input_prompt(user_input, metadata)
    # response = invoke_bedrock_claude(prompt)
    
    # embed= invoke_bedrock_titan(response[0])
    # print(embed)
    print(invoke_bedrock_claude("hello world"))

if __name__ == "__main__":
    main()
