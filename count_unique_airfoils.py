import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('airfoil_data.db')
cursor = conn.cursor()

# Query to count unique airfoil names
cursor.execute("SELECT COUNT(DISTINCT airfoilName) FROM airfoils")
unique_count = cursor.fetchone()[0]

print(f"Number of unique airfoil names: {unique_count}")

# Query to get all unique airfoil names
cursor.execute("SELECT DISTINCT airfoilName FROM airfoils ORDER BY airfoilName")
unique_names = cursor.fetchall()

# Write the names to a text file
with open('unique_airfoil_names.txt', 'w') as f:
    for name in unique_names:
        f.write(f"{name[0]}\n")

print(f"Unique airfoil names have been written to 'unique_airfoil_names.txt'")

# Close the connection
conn.close() 