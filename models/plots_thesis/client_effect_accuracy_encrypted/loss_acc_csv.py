import pandas as pd

# File paths (update these with the actual paths to your CSV files)
# paths = ['cfr_2cl_clients_acc_loss.csv', 'cfr_4cl_clients_acc_loss.csv', 'cfr_8cl_clients_acc_loss.csv']
paths = ['balnmp_2cl_loss_acc_clients.csv', 'balnmp_4cl_loss_acc_clients.csv', 'balnmp_8cl_loss_acc_clients.csv']

# Read the data and process each file
data_frames = []
for path in paths:
    # Read the CSV file
    df = pd.read_csv(path)

    # Remove 'MIN' and 'MAX' columns
    df = df.filter(regex='^(?!.*__MIN|.*__MAX)')

    # Rename 'Step' to 'Epoch' and create a sequence starting from 1
    df['Epoch'] = range(1, len(df) + 1)

    # Set 'Epoch' as the index
    df.set_index('Epoch', inplace=True)

    # Remove the original 'Step' column
    df.drop(columns=['Step'], inplace=True, errors='ignore')

    # Append the processed DataFrame to the list
    data_frames.append(df)

# Combine all DataFrames along the columns
combined_df = pd.concat(data_frames, axis=1)

# Display the combined DataFrame
print(combined_df.head())

combined_df.to_csv("loss_acc_clients_balnmp.csv")
