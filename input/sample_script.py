import pandas as pd

df = pd.DataFrame({
    'Age': [25, None, 30, 45, 50, None],
    'Income': [50000, 54000, None, 75000, 80000, 62000],
    'Gender': ['Male', 'Female', 'Female', 'Male', None, 'Female']
})

df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Income'].fillna(df['Income'].median(), inplace=True)
df['Gender'].fillna('Unknown', inplace=True)

df['Income'] = (df['Income'] - df['Income'].mean()) / df['Income'].std()

df = df[(df['Age'] >= 18) & (df['Age'] <= 65)]

print(df)
