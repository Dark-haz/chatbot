import pandas as pd
import json
from services import generate_embedding

import pandas as pd
import json
from services import generate_embedding

def embed_amazon_data(start_row, num_rows):
    csv_file_path = 'cleaned_filtered_dataset.csv'
    df = pd.read_csv(csv_file_path)

    df['combined_text'] = df[['product_name', 'price', 'amazon_category_and_sub_category', 'product_information']].astype(str).agg(' '.join, axis=1)

   
    combined_text_list = df['combined_text'].iloc[start_row:start_row+num_rows].tolist()

    
    result = []
    for i in range(len(combined_text_list)):
        rr = generate_embedding(combined_text_list[i])
        dic = {
            "description": combined_text_list[i],
            "embedding": rr,
            "itemId": df['uniq_id'].iloc[start_row + i] 
        }
        result.append(dic)

    # Write the result to a JSON file
    with open('embeddings_output.json', 'w') as json_file:
        json.dump(result, json_file, indent=4)

    return result

def main():
    # Set start_row and num_rows for the second 100 rows
    start_row = 100
    num_rows = 100
    rr = embed_amazon_data(start_row, num_rows)
    print(rr)

if __name__ == "__main__":
    main()


