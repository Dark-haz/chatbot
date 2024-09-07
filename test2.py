import pandas as pd
from services import generate_embedding
def embed_amazon_data():
    csv_file_path = 'cleaned_filtered_dataset.csv'
    df = pd.read_csv(csv_file_path)

    df['combined_text'] = df[['product_name', 'price', 'amazon_category_and_sub_category', 'product_information']].astype(str).agg(' '.join, axis=1)


    combined_text_list = df['combined_text'].head(100).tolist()

    result=[]
    for i in range(0,len(combined_text_list)):
        rr= generate_embedding(combined_text_list[i])
        dic={
            "description":combined_text_list[i],
            "embedding":rr,
            "itemId":df['uniq_id'][i]
        }
        result.append(dic)
    return result
def main():
    rr=embed_amazon_data()
    print(rr)
if __name__ =="__main__":
    main()
