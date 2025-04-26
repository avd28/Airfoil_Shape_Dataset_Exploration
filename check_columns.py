import sqlite3

# Connect to the database
conn = sqlite3.connect('airfoil_data.db')
cursor = conn.cursor()

# Get column names
cursor.execute("PRAGMA table_info(airfoils)")
columns = cursor.fetchall()

# Print column information
print("Column Information:")
print("==================")
for col in columns:
    print(f"Column {col[0]}: {col[1]} (Type: {col[2]})")

conn.close() 