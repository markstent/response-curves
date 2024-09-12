
# Advertising Response Curve Generator

![header](header.jpg)

This Streamlit app generates response curves for different advertising channels using a diminishing returns model and allows users to visualize and download the results as CSV files. The app processes a CSV file containing advertising spend data and a target variable, then generates response curves for each channel. This was created as an internal project that needed marketing response curves to model and optimize marketing budgets.

## Features

- Upload a CSV file with advertising spend data for multiple channels.
- Visualize the response curves for each channel based on a diminishing returns model.
- Download a combined CSV file containing the response curves data.
- Customizable logo and footer with attribution.

## Prerequisites

Before running the app locally, ensure you have the following installed:

- **Python 3.7+**
- **pip** (Python package installer)
  
You can download and install Python from [here](https://www.python.org/downloads/).

## Setup Instructions

1. **Clone the repository or download the app files**:
   - Place the following files in a folder:
     - `responsecurve.py` (the main Python script for the Streamlit app)
     - `spark_logo.png` (your custom logo file)

2. **Install the required Python packages**:
   - Open your terminal or command prompt and navigate to the folder where `responsecurve.py` is located.
   - Install the required packages by running the following command:

   ```bash
   pip install streamlit pandas matplotlib scipy scikit-learn
   ```

## Running the App Locally

1. **Navigate to the folder** where `responsecurve.py` is located:

   ```bash
   cd /path/to/your/app/folder
   ```

2. **Run the Streamlit app**:

   ```bash
   streamlit run app.py
   ```

3. **Open the app in your web browser**:
   - After running the `streamlit run` command, Streamlit will automatically launch the app in your default browser. If it doesn't open automatically, you can navigate to the URL provided in the terminal (usually `http://localhost:8501`).

## Expected CSV Format

Your CSV file should contain:

- **One `target` column**: This is the dependent variable you want to predict (e.g., sales, conversions).
- **Multiple advertising channels** with names starting with the prefix `s_`. Each channel's spend data should be in its own column. Examples:
  - `s_tv` (TV advertising spend)
  - `s_radio` (Radio advertising spend)
  - `s_newspaper` (Newspaper advertising spend)

### Example CSV Structure

```csv
date,s_tv,s_radio,s_newspaper,target
2023-01-01,230.1,37.8,69.2,22.1
2023-01-08,44.5,39.3,45.1,10.4
2023-01-15,17.2,45.9,69.3,9.3
...
```

Dummy data can be found [here](Advertising_Weekly_Dates.csv)

## Customization

- **Logo**: You can customize the logo displayed at the top by replacing the `spark_logo.png` file with your own logo file.
- **Footer**: The app displays "Developed by Mark Stent" at the bottom. You can change this message or remove it by modifying the relevant section in `app.py`.

## License

This project is open source and available under the MIT License.
