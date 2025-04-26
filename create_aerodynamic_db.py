import sqlite3
import pandas as pd

# Connect to the original database
conn_original = sqlite3.connect('airfoil_data.db')

# Define which columns to keep (aerodynamic parameters)
keep_columns = [
    'airfoilName',  # Identifier
    'coefficientLift', 'coefficientDrag', 'coefficientMoment', 'coefficientParasiteDrag',  # Coefficient columns
    'reynoldsNumber',  # Flow parameter
    'topXTR', 'botXTR'  # Transition points
]

# Create the query to select only the desired columns
columns_str = ', '.join(keep_columns)
query = f"SELECT DISTINCT {columns_str} FROM airfoils"

# Read the data into a DataFrame
df = pd.read_sql_query(query, conn_original)

# Create a new database
conn_new = sqlite3.connect('airfoil_aero_params.db')

# Write the data to the new database
df.to_sql('airfoils', conn_new, if_exists='replace', index=False)

# Close connections
conn_original.close()
conn_new.close()

print("New database 'airfoil_aero_params.db' created successfully!")
print(f"Number of unique records in new database: {len(df)}")
print("Columns in new database:", keep_columns) 