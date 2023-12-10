import pandas as pd

# Load CSV data
# csv_path = 'cfr_epoc_round_loss_acc.csv'  # Update with your actual CSV file path
csv_path = 'files/balnmp_2cl_allexp_50r_1e_loss_acc.csv'
df = pd.read_csv(csv_path)
print(df.head())
# Remove columns containing '__MIN' and '__MAX'
df = df.loc[:, ~df.columns.str.contains('__MIN|__MAX')]

# Convert 'Step' to sequential epochs, assuming steps start at 1 and increment by 1
df['Epoch'] = (df.index) + 1  # Assuming each row corresponds to a new epoch

# Drop the original 'Step' column
df.drop(columns='Step', inplace=True)

# Save the updated CSV file
updated_csv_path = 'updated_balnmp_50r1e_loss_acc.csv'
df.to_csv(updated_csv_path, index=False)

print(f"Updated CSV saved to {updated_csv_path}")
