# Data_analysis_ml_methods_autoencoder
This code loads network data, preprocesses it, reduces dimensions with an autoencoder, and trains multiple classifiers (KNN, RF, LR, SVM) for anomaly detection.

----
Sure, here's a sample README file for the provided Python code:

---

# Network Anomaly Detection

This project implements a machine learning pipeline to detect network anomalies using various classification algorithms. The dataset used for this project contains network traffic data with labeled anomalies.

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Project Structure](#project-structure)
4. [Data](#data)
5. [Exploratory Data Analysis](#exploratory-data-analysis)
6. [Data Preprocessing](#data-preprocessing)
7. [Dimensionality Reduction](#dimensionality-reduction)
8. [Model Training and Evaluation](#model-training-and-evaluation)
9. [Results](#results)
10. [Contributing](#contributing)
11. [License](#license)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/HayatiYrtgl/Data_analysis_ml_methods_autoencoder.git
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. Ensure your dataset is placed in the appropriate directory, as specified in the code (`../dataset/network_anomaly_detection/all_data (3).csv`).

2. Run the Jupyter notebook:
    ```bash
    jupyter notebook
    ```
   Open the notebook and execute the cells to run the entire pipeline.

## Project Structure

```
network-anomaly-detection/
├── dataset/
│   └── network_anomaly_detection/
│       └── all_data (3).csv
├── README.md
├── requirements.txt
└── anomaly_detection.ipynb
```

- `dataset/`: Directory containing the dataset.
- `README.md`: This file.
- `requirements.txt`: List of Python packages required for the project.
- `anomaly_detection.ipynb`: Jupyter notebook containing the code.

## Data

The dataset used for this project contains network traffic data with labeled anomalies. It is loaded from a CSV file located in the `dataset/network_anomaly_detection/` directory.

## Exploratory Data Analysis

Initial data exploration includes:
- Viewing the first and last few rows of the dataset.
- Checking data types and missing values.
- Plotting the distribution of the target variable.
- Visualizing feature correlations.

## Data Preprocessing

Steps include:
- Handling missing values and duplicates.
- Encoding categorical variables.
- Scaling numerical features using MinMaxScaler.

## Dimensionality Reduction

An autoencoder is used for dimensionality reduction to select the most important features.

## Model Training and Evaluation

Four different classifiers are trained and evaluated:
- K-Nearest Neighbors (KNN)
- Random Forest
- Logistic Regression
- Support Vector Machine (SVM)

The performance of each model is assessed using a classification report.

## Results

The results of the models, including precision, recall, and F1-score, are printed for each classifier.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

