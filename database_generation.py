import pandas as pd
import sqlite3

# Adjust the chunk size as needed for your memory
chunksize = 10000
csv_file = 'combinedAirfoilDataLabeled.csv'
db_file = 'airfoil_data.db'
table_name = 'airfoils'

# Create a connection to the SQLite database
conn = sqlite3.connect(db_file)

# Read and write in chunks
for chunk in pd.read_csv(csv_file, chunksize=chunksize):
    chunk.to_sql(table_name, conn, if_exists='append', index=False)

conn.close()
print("CSV has been converted to SQLite database:", db_file)