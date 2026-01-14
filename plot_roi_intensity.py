import matplotlib.pyplot as plt


def extract_and_plot_roi_intensity(data):
    """
    Extract ROIs for each plane and plot their intensity along trial number.

    Args:
        data (dict): A dictionary containing planes, trials, and ROI data.
    """
    for plane_id, plane_data in data.items():
        trial_numbers = []
        roi_intensities = []

        for trial_number, trial in enumerate(plane_data['trials']):
            trial_numbers.append(trial_number)
            roi_intensities.append(trial['cs_us_vs_pre'])  # Extract ROI intensity

        # Plotting
        plt.figure()
        plt.plot(trial_numbers, roi_intensities, marker='o', label=f'Plane {plane_id}')
        plt.xlabel('Trial Number')
        plt.ylabel('ROI Intensity (cs_us_vs_pre)')
        plt.title(f'ROI Intensity Across Trials for Plane {plane_id}')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage
# Replace `data` with your actual data structure
data = {
    "plane_1": {
        "trials": [
            {"cs_us_vs_pre": 0.5},
            {"cs_us_vs_pre": 0.7},
            {"cs_us_vs_pre": 0.6},
        ]
    },
    "plane_2": {
        "trials": [
            {"cs_us_vs_pre": 0.4},
            {"cs_us_vs_pre": 0.8},
            {"cs_us_vs_pre": 0.9},
        ]
    }
}

extract_and_plot_roi_intensity(data)
