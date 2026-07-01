import pandas as pd

df = pd.read_csv('data.csv')
# print(df.isnull().sum())

df['Calories'] = df['Calories'].fillna(df['Calories'].mean())

grouped_data = df.groupby('Duration').mean()

print("--- Average Pulse, Maxpulse and Calories for each Duration ---")
print(grouped_data)