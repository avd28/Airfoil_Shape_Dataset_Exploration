import sqlite3
import numpy as np
import matplotlib.pyplot as plt

def generate_airfoil_coordinates(airfoil_name, db_file='airfoil_data.db', table_name='airfoils', n_points=200, output_dir='.', show_plot=True):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Query for the airfoil row using the correct column name
    cursor.execute(f"SELECT * FROM {table_name} WHERE airfoilName = ?", (airfoil_name,))
    row = cursor.fetchone()
    if not row:
        print(f"Airfoil '{airfoil_name}' not found in the database.")
        conn.close()
        return None, None, None

    # Get column names to find coefficient indices
    col_names = [desc[0] for desc in cursor.description]
    
    # Find indices for upper and lower coefficients using the correct naming convention
    # Now coefficients will be in correct order: Coeff1 goes with x^30, Coeff2 with x^29, etc.
    upper_coeffs = [row[col_names.index(f'upperSurfaceCoeff{i}')] for i in range(1, 32)]
    lower_coeffs = [row[col_names.index(f'lowerSurfaceCoeff{i}')] for i in range(1, 32)]

    # Generate x values from 0 to 1
    x = np.linspace(0, 1, n_points)
    
    # Evaluate polynomials - no need to reverse coefficients now as they're already in the right order
    # upperSurfaceCoeff1 * x^30 + upperSurfaceCoeff2 * x^29 + ... + upperSurfaceCoeff31 * x^0
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

    if show_plot:
        plot_airfoil(x, y_upper, y_lower, airfoil_name)

    return x, y_upper, y_lower

def plot_airfoil(x, y_upper, y_lower, airfoil_name):
    """
    Plot the airfoil coordinates with proper scaling and formatting.
    
    Args:
        x (numpy.array): x-coordinates
        y_upper (numpy.array): y-coordinates of upper surface
        y_lower (numpy.array): y-coordinates of lower surface
        airfoil_name (str): name of the airfoil for the plot title
    """
    plt.figure(figsize=(10, 5))
    
    # Plot upper and lower surfaces
    plt.plot(x, y_upper, 'b-', label='Upper Surface', linewidth=2)
    plt.plot(x, y_lower, 'r-', label='Lower Surface', linewidth=2)
    
    # Fill the airfoil
    plt.fill_between(x, y_upper, y_lower, alpha=0.1, color='gray')
    
    # Set equal aspect ratio to preserve shape
    plt.axis('equal')
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add labels and title
    plt.xlabel('x/c')
    plt.ylabel('y/c')
    plt.title(f'Airfoil Shape: {airfoil_name}')
    plt.legend()
    
    # Adjust plot margins
    plt.margins(x=0.1)
    
    # Show the plot
    plt.show()

def plot_airfoil_from_file(filename):
    """
    Plot airfoil coordinates from a previously generated coordinate file.
    
    Args:
        filename (str): path to the coordinate file
    """
    # Read the coordinates
    data = np.loadtxt(filename, skiprows=1)
    x = data[:, 0]
    y_upper = data[:, 1]
    y_lower = data[:, 2]
    
    # Extract airfoil name from filename
    airfoil_name = filename.split('/')[-1].replace('_coordinates.txt', '')
    
    # Plot the airfoil
    plot_airfoil(x, y_upper, y_lower, airfoil_name)

# Example usage:
if __name__ == "__main__":
    # Generate and plot coordinates for a specific airfoil
    airfoil_name = "2032c"  # Replace with an actual airfoil name from your database
    
    # Method 1: Generate coordinates and plot directly
    x, y_upper, y_lower = generate_airfoil_coordinates(airfoil_name, show_plot=True)
    
    # Method 2: Plot from a previously generated file
    # plot_airfoil_from_file(f"{airfoil_name}_coordinates.txt")