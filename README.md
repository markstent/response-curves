
# Advertising Response Curve Generator

![header](header.jpg)

This Streamlit app generates response curves for different advertising channels using a diminishing returns model and allows users to visualize and download the results as CSV files. The app now includes an Adstock model to account for the carryover effects of advertising spend across time, allowing users to fine-tune the Adstock decay rate for each channel. This was created as an internal project that needed marketing response curves to model and optimize marketing budgets.

## Features

 - Upload a CSV file with advertising spend data for multiple channels.
 - Adjust Adstock decay rates for each channel using sliders.
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
     - `spark-logo.png` (your custom logo file)

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

- **Logo**: You can customize the logo displayed at the top by replacing the `spark-logo.png` file with your own logo file.
- **Footer**: The app displays "Developed by Mark Stent" at the bottom. You can change this message or remove it by modifying the relevant section in `responsecurve.py`.

## Adstock Parameters

- The app allows users to adjust the Adstock decay rate for each channel using sliders. Adstock models the carryover effect of - advertising, simulating how advertising in one period can affect future periods.
- Each channel has a default decay rate of 0.5, which can be adjusted between 0.0 (no carryover) to 1.0 (full carryover).

## Attribution Methodology and Curve Generation

1. **Adstock Transformation:

- For each channel, apply an Adstock transformation using the decay rate set by the user. This models the carryover effect of advertising spend over time.

2. **Multivariate Regression for Attribution**:
   - Perform a **multivariate linear regression** using advertising spend columns (`s_` columns) as independent variables and the `target` column as the dependent variable.
   - The regression outputs **coefficients** for each channel, representing the contribution of each channel to the target variable.

3. **Scaling by Coefficients**:
   - Each channel's contribution to the target is **scaled** by its respective regression coefficient, ensuring the target variable's sensitivity to each channel's spend is properly adjusted.

4. **Define Common X-Range (Advertising Spend)**:
   - Establish a **common x-range** (spend range) from 0 to the maximum observed spend across all channels.
   - This ensures all channels' response curves are calculated over a consistent range of spend values.

5. **Fit the Diminishing Returns Model**:
   - For each channel, apply a **logarithmic diminishing returns model**:

    $f(x) = a \cdot \log(1 + b \cdot x)$

     - Estimate the parameters \( a \) and \( b \) using curve fitting, which shapes the curve to reflect diminishing returns as spend increases.

6. **Generate Predicted Target Values**:
   - Use the fitted model to compute **predicted target values** for each channel over the common x-range (spend values), showing how the target variable (e.g., sales) changes with increasing spend for each channel.

## License

This project is open source and available under the MIT License.
