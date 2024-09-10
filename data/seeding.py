from sqlalchemy import create_engine , text
from config.static import *
from Models.Item import Item

## SEEDING  ---------------------------------------------------
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import random

import boto3
import json
from config.static import *


def get_bedrock_client():
    """Initialize and return the Bedrock client."""
    return boto3.client(service_name="bedrock-runtime", region_name='us-east-1')

def generate_embedding(prompt_data):
    modelId = "amazon.titan-embed-text-v2:0"
    accept = "application/json"
    contentType = "application/json"

    
    sample_model_input = {
        "inputText": prompt_data,
        "dimensions": VECTOR_DIMENSION,
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


def generate_random_vector(length):
    return [random.uniform(0.0, 1.0) for _ in range(length)]


engine = create_engine(DATABASE_URL)

Session = sessionmaker(bind=engine)
session = Session()

# seed_data = [
#     Item(itemId=1, description="Wireless Mouse with ergonomic design and customizable buttons for gaming or office use.", embedding=generate_random_vector(VECTOR_DIMENSION)),
#     Item(itemId=2, description="High-performance Laptop with a 15.6-inch display, Intel i7 processor, 16GB RAM, and 512GB SSD.", embedding=generate_random_vector(VECTOR_DIMENSION)),
#     Item(itemId=3, description="Comfortable and durable jeans, perfect for casual wear. Available in various sizes and colors.", embedding=generate_random_vector(VECTOR_DIMENSION)),
#     Item(itemId=4, description="Stylish baseball cap with adjustable strap. Made of 100% cotton for a comfortable fit.", embedding=generate_random_vector(VECTOR_DIMENSION)),
#     Item(itemId=5, description="High-quality ballpoint pen with smooth ink flow. Ideal for everyday writing tasks.", embedding=generate_random_vector(VECTOR_DIMENSION)),
#     Item(itemId=6, description="Sleek and lightweight laptop with a 14-inch display, AMD Ryzen 5 processor, 8GB RAM, and 256GB SSD.", embedding=generate_random_vector(VECTOR_DIMENSION)),
#     Item(itemId=7, description="Premium laptop with a 17.3-inch 4K display, Intel i9 processor, 32GB RAM, and 1TB SSD. Ideal for gaming and professional work.", embedding=generate_random_vector(VECTOR_DIMENSION)),
# ]

seed_data = [
    Item(itemId=1, description="Wireless Mouse with ergonomic design and customizable buttons for gaming or office use.", embedding=generate_embedding("Wireless Mouse with ergonomic design and customizable buttons for gaming or office use.")),
    Item(itemId=2, description="High-performance Laptop with a 15.6-inch display, Intel i7 processor, 16GB RAM, and 512GB SSD.", embedding=generate_embedding("High-performance Laptop with a 15.6-inch display, Intel i7 processor, 16GB RAM, and 512GB SSD.")),
    Item(itemId=3, description="Comfortable and durable jeans, perfect for casual wear. Available in various sizes and colors.", embedding=generate_embedding("Comfortable and durable jeans, perfect for casual wear. Available in various sizes and colors.")),
    Item(itemId=4, description="Stylish baseball cap with adjustable strap. Made of 100% cotton for a comfortable fit.", embedding=generate_embedding("Stylish baseball cap with adjustable strap. Made of 100% cotton for a comfortable fit.")),
    Item(itemId=5, description="High-quality ballpoint pen with smooth ink flow. Ideal for everyday writing tasks.", embedding=generate_embedding("High-quality ballpoint pen with smooth ink flow. Ideal for everyday writing tasks.")),
    Item(itemId=6, description="Sleek and lightweight laptop with a 14-inch display, AMD Ryzen 5 processor, 8GB RAM, and 256GB SSD.", embedding=generate_embedding("Sleek and lightweight laptop with a 14-inch display, AMD Ryzen 5 processor, 8GB RAM, and 256GB SSD.")),
    Item(itemId=7, description="Premium laptop with a 17.3-inch 4K display, Intel i9 processor, 32GB RAM, and 1TB SSD. Ideal for gaming and professional work.", embedding=generate_embedding("Premium laptop with a 17.3-inch 4K display, Intel i9 processor, 32GB RAM, and 1TB SSD. Ideal for gaming and professional work."))
]





session.query(Item).delete()
session.execute(text("SELECT setval(pg_get_serial_sequence('\"Items\"', 'id'), 1, false)"))



session.add_all(seed_data)
session.commit()
session.close()