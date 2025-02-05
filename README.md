# Perceptron Learning Algorithm (PLA) Visualization

## Overview
This Streamlit application visualizes the **Perceptron Learning Algorithm (PLA)** using interactive plots. The app showcases how the Perceptron model learns to classify points into two classes by iteratively updating its weights based on misclassifications. The visualizations cover three types of datasets:
1. **Linearly Separable Dataset**
2. **Non-Linear Dataset (e.g., Circles)**
3. **Noisy Dataset**

The user can interactively train the Perceptron model on these datasets and visualize the learning process, including decision boundaries and loss curves over epochs.

## Requirements
To run the application, you need the following libraries:
- `streamlit`
- `numpy`
- `matplotlib`
- `scikit-learn`

You can install the required libraries using pip:
```bash
pip install streamlit numpy matplotlib scikit-learn
```

## How It Works
The `PerceptronVisualizer` class is at the core of this app. It handles the following tasks:
1. **Training the Perceptron**: It initializes the model's weights and bias, and iteratively updates them using the Perceptron learning rule based on the classification errors (misclassifications).
2. **Visualizing the Decision Boundary**: The decision boundary is animated over training epochs, showing how the model's separation line evolves.
3. **Plotting the Loss Curve**: A plot displays how the number of misclassified points decreases over the epochs.

### Perceptron Update Rule
- If a data point is misclassified, the weights are updated as follows:
  ```
  w = w + η * y * x
  bias = bias + η * y
  ```
  Where:
  - `w` is the weight vector.
  - `η` is the learning rate.
  - `y` is the true label of the data point.
  - `x` is the feature vector of the data point.
  - `bias` is the bias term.

## Datasets
The app includes three types of datasets for classification:
1. **Linearly Separable**: A dataset with linearly separable classes.
2. **Non-Linear**: A dataset created using `make_circles`, which is not linearly separable.
3. **Noisy**: A dataset with added noise (`flip_y=0.1`) to simulate real-world scenarios where misclassifications are common.

## Features
- **Interactive Controls**: 
  - **Max Epochs**: Slider to control the maximum number of epochs for training.
  - **Train Button**: Button to start the training process.
  
- **Visualizations**:
  - **Animated Decision Boundary**: A dynamic plot showing how the decision boundary evolves with each epoch.
  - **Loss Curve**: A plot showing the number of misclassified points in each epoch.

## Usage
To run the application:
1. Save the script as `app.py`.
2. Run the Streamlit app using the command:
   ```bash
   streamlit run app.py
   ```
3. Open the web interface and interact with the visualizations.
