import pandas as pd

# Path to your Parquet file
file_path = "outputs_new_prompt/combined_embeddings.parquet"
file_other = "outputs_new_prompt/combined_articles.csv"
# Load the Parquet file into a DataFrame
df = pd.read_parquet(file_path)
el = pd.read_csv(file_other)
# Print column names
print("Column Names:")
print(df.columns.tolist())

# Print data types of each column
print("\nData Types:")
print(df.dtypes)

print("Article column names")
print (el.columns.tolist())
print("\nData Types:")
print(el.dtypes)
