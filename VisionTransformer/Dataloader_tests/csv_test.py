# import pandas as pd

# # Read the CSV file
# csv_file_path = "/workspace/arijit/sandipan/zsgr_caddy/hariansh/raw_dataset_caddy/unseen_img_label.csv"
# df = pd.read_csv(csv_file_path)

# # Count the occurrences of each unique label
# label_counts = df['label id'].value_counts()

# # Print the number of entries for each label
# print("Number of entries for each label:")
# print(label_counts)

import pandas as pd

# Load the CSV file into a DataFrame
df = pd.read_csv('/workspace/arijit/sandipan/zsgr_caddy/hariansh/raw_dataset_caddy/seen_img_label.csv')

# Replace all occurrences of -1 in the 'label id' column with 16
df.loc[df['label id'] == -1, 'label id'] = 16

# Save the modified DataFrame back to the same CSV file
df.to_csv('/workspace/arijit/sandipan/zsgr_caddy/hariansh/raw_dataset_caddy/seen_img_label.csv', index=False)