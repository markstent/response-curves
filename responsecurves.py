import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from io import StringIO
from matplotlib.ticker import FuncFormatter


def thousands_separator(x, pos):
    """The two arguments are the value and tick position."""
    return f'{x:,.0f}'


class ResponseCurveApp:
    def __init__(self):
        self.df = None
        self.curve_points = None
        self.x_range = None

    # Adstock transformation function
    @staticmethod
    def apply_adstock(spend, decay_rate):
        adstock_spend = np.zeros_like(spend)
        for t in range(1, len(spend)):
            adstock_spend[t] = spend[t] + decay_rate * adstock_spend[t - 1]
        return adstock_spend

    # Diminishing returns function (nonlinear model)
    @staticmethod
    def diminishing_returns(x, a, b):
        return a * np.log(1 + b * x)

    # Objective function to optimize the adstock decay rate
    def optimize_adstock_decay(self, spend, target, initial_decay=0.5):
        def objective_function(params):
            decay_rate = params[0]
            a, b = params[1], params[2]

            # Apply Adstock
            adstock_spend = self.apply_adstock(spend, decay_rate)

            # Predicted target with diminishing returns
            predicted = self.diminishing_returns(adstock_spend, a, b)

            # Minimize mean squared error between actual and predicted
            return mean_squared_error(target, predicted)

        # Initial guess for [decay_rate, a, b]
        initial_params = [initial_decay, 1, 0.1]

        # Perform optimization
        result = minimize(objective_function, x0=initial_params, bounds=[(0, 1), (0, None), (0, None)])
        return result.x[0], result.x[1], result.x[2]  # Return decay rate and parameters a, b

    # Load and validate the dataset
    def load_data(self, uploaded_file):
        try:
            # Load the CSV into a pandas DataFrame
            self.df = pd.read_csv(uploaded_file)
            st.write("Dataset Preview:")
            st.write(self.df.head())

            # Check if the dataset has 'target' and any 's_' prefixed columns
            if 'target' not in self.df.columns:
                st.error("The dataset must have a 'target' column.")
                return False

            channel_columns = [col for col in self.df.columns if col.startswith('s_')]
            if not channel_columns:
                st.error("The dataset must have at least one column starting with 's_' as a channel.")
                return False

            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False

    # Function to generate response curve points for each channel, incorporating adstock
    def generate_response_curves(self, decay_rates, params):
        channel_columns = [col for col in self.df.columns if col.startswith('s_')]
        y = self.df['target']
        X = self.df[channel_columns]

        self.curve_points = {}

        # Define a common range for spend (from the minimum to the maximum spend across all channels)
        self.x_range = np.linspace(0, max(self.df[channel_columns].max()), 100)

        # For each channel, compute response curve points
        for i, channel in enumerate(channel_columns):
            X_channel = self.df[channel].values

            # Apply Adstock transformation with the optimal decay rate
            adstock_spend = self.apply_adstock(X_channel, decay_rates[channel])

            # Use the diminishing returns parameters
            a, b = params[channel]

            # Predicted target using diminishing returns
            y_pred = self.diminishing_returns(self.x_range, a, b)

            # Store the curve points
            self.curve_points[channel] = pd.DataFrame({
                'spend': self.x_range,
                f'{channel}_predicted_target': y_pred
            })

    # Function to visualize all curves on one graph using Seaborn
    def visualize_all_curves(self):
        if not self.curve_points:
            st.error("No curve points available to visualize.")
            return

        # Use Seaborn's style for a cleaner look
        sns.set(style="whitegrid")

        plt.figure(figsize=(10, 6))

        # Plot each channel's curve with seaborn lineplot
        for channel, points in self.curve_points.items():
            sns.lineplot(x=points['spend'], y=points[f'{channel}_predicted_target'], label=channel)

        # Set axis labels and title
        plt.title('Response Curves for All Advertising Channels (with Adstock and Diminishing Returns)')
        plt.xlabel('Spend')
        plt.ylabel('Predicted Target')

        # Format both x-axis and y-axis ticks with a thousands separator
        plt.gca().xaxis.set_major_formatter(FuncFormatter(thousands_separator))
        plt.gca().yaxis.set_major_formatter(FuncFormatter(thousands_separator))

        # Display legend and grid
        plt.legend()
        plt.grid(True)

        # Use Streamlit to render the plot
        st.pyplot(plt)

    # Function to convert all curve points into a single CSV format with a single spend column
    def convert_to_single_csv(self):
        if not self.curve_points:
            st.error("No curve points available for download.")
            return None

        # Start with the first channel's data and merge subsequent channel data on 'spend'
        combined_df = pd.DataFrame({'spend': self.x_range})

        for channel, points in self.curve_points.items():
            combined_df = pd.merge(combined_df, points, on='spend', how='left')

        csv_buffer = StringIO()
        combined_df.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()

    # Display curve data
    def display_curve_data(self):
        if not self.curve_points:
            st.error("No curve points available to display.")
            return None

        # Start with the first channel's data and merge subsequent channel data on 'spend'
        combined_df = pd.DataFrame({'spend': self.x_range})

        for channel, points in self.curve_points.items():
            combined_df = pd.merge(combined_df, points, on='spend', how='left')

        st.write("Curve Data:")
        st.dataframe(combined_df)
        return combined_df

    # Display optimal parameters in a table format
    def display_optimal_parameters(self, optimal_decay_rates, params):
        # Create a DataFrame to hold the parameters
        param_data = {
            'Channel': list(optimal_decay_rates.keys()),
            'Decay Rate': [f"{decay:.2f}" for decay in optimal_decay_rates.values()],
            'a': [f"{params[channel][0]:.2f}" for channel in optimal_decay_rates.keys()],
            'b': [f"{params[channel][1]:.2f}" for channel in optimal_decay_rates.keys()]
        }

        # Convert to a pandas DataFrame
        param_df = pd.DataFrame(param_data)

        # Display the table in Streamlit
        st.write("**Optimal Adstock Decay Rates and Parameters (a, b)**")
        st.table(param_df)


# Streamlit app class that runs the flow
class StreamlitApp:
    def __init__(self):
        self.response_curve_app = ResponseCurveApp()

    def run(self):
        # Display the logo
        st.image("spark-logo.png", width=150)

        st.title("Advertising Response Curve Generator (with Adstock and Diminishing Returns)")

        st.markdown("""
        This app generates response curves for different advertising channels using diminishing returns and allows you to visualize and download the results. It also incorporates Adstock for carryover effects.

        **Expected CSV format**:
        - Columns must include:
          - One `target` column (dependent variable)
          - Multiple advertising channels with names starting with `s_` (e.g., `s_tv`, `s_radio`)
        """)

        # File upload
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

        if uploaded_file is not None:
            # Load and validate data
            if self.response_curve_app.load_data(uploaded_file):
                channel_columns = [col for col in self.response_curve_app.df.columns if col.startswith('s_')]

                # Optimize decay rates and parameters
                optimal_decay_rates = {}
                params = {}
                for channel in channel_columns:
                    optimal_decay, a, b = self.response_curve_app.optimize_adstock_decay(
                        self.response_curve_app.df[channel],
                        self.response_curve_app.df['target']
                    )
                    optimal_decay_rates[channel] = optimal_decay
                    params[channel] = (a, b)

                # Display optimal decay rates and parameters in a table
                self.response_curve_app.display_optimal_parameters(optimal_decay_rates, params)

                # Set adstock decay rates for each channel dynamically with optimal defaults
                decay_rates = {}
                st.write("**Adstock Parameters:**")
                for channel in channel_columns:
                    decay_rates[channel] = st.slider(
                        f"Adstock Decay Rate for {channel}",
                        0.0, 1.0, optimal_decay_rates[channel]  # Use the optimized decay rate as the default
                    )

                # Generate response curves using the selected adstock decay rates
                self.response_curve_app.generate_response_curves(decay_rates, params)

                # Visualize all curves on a single graph
                st.write("**Response Curves:**")
                self.response_curve_app.visualize_all_curves()

                # Display the curve data
                all_data = self.response_curve_app.display_curve_data()

                # Option to download all curve points as a single CSV
                if all_data is not None:
                    st.write("Download the combined curve points as a CSV file:")
                    csv_data = self.response_curve_app.convert_to_single_csv()
                    if csv_data:
                        st.download_button(
                            label="Download CSV",
                            data=csv_data,
                            file_name="all_curve_points.csv",
                            mime="text/csv"
                        )

        # Add text at the bottom
        st.markdown("<br><hr><center>Developed by Mark Stent</center>", unsafe_allow_html=True)


# Run the Streamlit app
if __name__ == "__main__":
    app = StreamlitApp()
    app.run()
