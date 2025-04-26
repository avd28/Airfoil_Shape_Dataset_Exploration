# Airfoil Performance Dashboard

A comprehensive web-based dashboard for analyzing and comparing airfoil performance characteristics using Streamlit and Plotly.

## Overview

This dashboard provides an interactive interface to explore and analyze airfoil performance data, including lift coefficients, drag coefficients, and moment coefficients across different Reynolds numbers and angles of attack. The application is built using Streamlit and features dynamic visualizations powered by Plotly.

## Features

- **Individual Airfoil Analysis**
  - Detailed performance metrics for selected airfoils
  - Interactive lift vs. drag polar plots
  - Lift and drag coefficient vs. angle of attack curves
  - Performance statistics including maximum lift and minimum drag coefficients

- **Airfoil Comparison**
  - Side-by-side comparison of up to 5 airfoils
  - Multiple visualization types:
    - Lift vs. Drag polar plots
    - Lift vs. Angle of Attack
    - Drag vs. Angle of Attack
  - Performance metrics comparison for selected airfoils

- **Interactive Features**
  - Real-time data filtering
  - Dynamic plot updates
  - Hover tooltips with detailed information
  - Customizable Reynolds number selection

## Technical Stack

- **Frontend**: Streamlit
- **Data Visualization**: Plotly
- **Data Management**: SQLite
- **Data Processing**: Pandas

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd Airfoil_Shape_Dataset_Exploration
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Ensure you have the SQLite database (`airfoil_data.db`) in the project directory
2. Run the Streamlit app:
```bash
streamlit run airfoil_dashboard.py
```

3. Access the dashboard through your web browser at the provided local URL (typically http://localhost:8501)

## Data Structure

The application expects a SQLite database with the following schema:

```sql
CREATE TABLE airfoils (
    airfoilName TEXT,
    coefficientLift REAL,
    coefficientDrag REAL,
    coefficientMoment REAL,
    reynoldsNumber REAL,
    alpha REAL
);
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify your license here]

## Contact

[Your contact information]
