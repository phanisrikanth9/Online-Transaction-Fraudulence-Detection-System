import pandas as pd
from sklearn.model_selection import train_test_split

# Load your dataset
df = pd.read_csv(r'c:\Users\SRIKANTH\Desktop\New folder (2)\archive (2)\PS_20174392719_1491204439457_log.csv')

# Use 'isFraud' as the target column for stratification
target = 'isFraud'

# First split
df_part1, df_temp = train_test_split(df, test_size=0.67, stratify=df[target], random_state=42)

# Second split
df_part2, df_part3 = train_test_split(df_temp, test_size=0.5, stratify=df_temp[target], random_state=42)

# Save the splits to separate files
df_part1.to_csv('dataset_part1.csv', index=False)
df_part2.to_csv('dataset_part2.csv', index=False)
df_part3.to_csv('dataset_part3.csv', index=False)

