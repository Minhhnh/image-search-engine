import pandas as pd

df = pd.read_csv("word_segmented.csv")['image']
df = pd.DataFrame(list(df[::5]))
df.columns = ['image']
print(df)
df.to_csv('images.csv', index=False)
