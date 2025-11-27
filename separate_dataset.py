import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("dataset_raw.csv")

train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=42)
val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=42)

train.to_csv("train.csv", index=False)
val.to_csv("val.csv", index=False)
test.to_csv("test.csv", index=False)

print("Train:", len(train))
print("Val:", len(val))
print("Test:", len(test))
