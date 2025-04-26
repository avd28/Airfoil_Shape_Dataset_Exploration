import sqlite3
import numpy as np

def generate_airfoil_coordinates(airfoil_name, db_file='airfoil_data.db', table_name='airfoils', n_points=200, output_dir='.'):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Query for the airfoil row using the correct column name
    cursor.execute(f"SELECT * FROM {table_name} WHERE airfoilName = ?", (airfoil_name,))
    row = cursor.fetchone()
    if not row:
        print(f"Airfoil '{airfoil_name}' not found in the database.")
        conn.close()
        return

    # Get column names to find coefficient indices
    col_names = [desc[0] for desc in cursor.description]
    # Find indices for upper and lower coefficients
    upper_coeffs = [row[col_names.index(f'upper_{i}')] for i in range(30)]
    lower_coeffs = [row[col_names.index(f'lower_{i}')] for i in range(30)]

    # Generate x values from 0 to 1
    x = np.linspace(0, 1, n_points)
    # Evaluate 30th order polynomial for upper and lower surfaces
    y_upper = np.polyval(upper_coeffs, x)
    y_lower = np.polyval(lower_coeffs, x)

    # Write to a text file
    output_file = f"{output_dir}/{airfoil_name}_coordinates.txt"
    with open(output_file, 'w') as f:
        f.write("# x y_upper y_lower\n")
        for xi, yu, yl in zip(x, y_upper, y_lower):
            f.write(f"{xi:.6f} {yu:.6f} {yl:.6f}\n")

    print(f"Coordinates written to {output_file}")
    conn.close()

# Example usage:
# generate_airfoil_coordinates('NACA2412')