import pandas as pd
import json
from services import generate_embedding

def embed_amazon_data():
    csv_file_path = 'cleaned_filtered_dataset.csv'
    df = pd.read_csv(csv_file_path)

    # Combine text fields into one column
    df['combined_text'] = df[['product_name', 'price', 'amazon_category_and_sub_category', 'product_information']].astype(str).agg(' '.join, axis=1)

    # Select top 100 combined texts
    combined_text_list = df['combined_text'].head(100).tolist()

    # Generate embeddings and prepare result list
    result = []
    for i in range(len(combined_text_list)):
        rr = generate_embedding(combined_text_list[i])
        dic = {
            "description": combined_text_list[i],
            "embedding": rr,
            "itemId": df['uniq_id'].iloc[i]  # Use iloc to ensure the correct indexing
        }
        result.append(dic)

    # Write the result to a JSON file
    with open('embeddings_output.json', 'w') as json_file:
        json.dump(result, json_file, indent=4)

    return result

def main():
    rr=embed_amazon_data()
    print(rr)
if __name__ =="__main__":
    main()
