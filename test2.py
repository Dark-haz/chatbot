import pandas as pd
from services import generate_embedding
csv_file_path = 'cleaned_filtered_dataset.csv'
df = pd.read_csv(csv_file_path)

df['combined_text'] = df[['product_name', 'price', 'amazon_category_and_sub_category', 'product_information']].astype(str).agg(' '.join, axis=1)


combined_text_list = df['combined_text'].head(2).tolist()

result=[]
for i in range(0,len(combined_text_list)):
   rr= generate_embedding(combined_text_list[i])
   dic={
      "description":combined_text_list[i],
      "vector":rr,
      "id":df['uniq_id'][i]
   }
   result.append(dic)
print(result)