import pandas as pd


"""
This script is ...
"""
filepath = "data/training-dataset.csv"
df = pd.read_csv(filepath,names=['sentiment', 'tweet_text'], delimiter=',')
# delete the first row "sentiment" and "tweet_text"
df = df.drop(0)
df.insert(0, 'id', range(len(df)), allow_duplicates=False)
df = df.rename(columns={'sentiment':'category', 'tweet_text':'text'})
df.to_csv('data/Black_dataset.csv', header=False)