import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles

class PerceptronVisualizer:
    def __init__(self, X, y, learning_rate=0.01):
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.history = []
        self.loss_history = []
        
    def initialize_weights(self):
        self.weights = np.random.randn(self.X.shape[1]) * 0.01
        self.bias = np.random.randn() * 0.01
        
    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.bias)
    
    def train(self, max_epochs):
        self.initialize_weights()
        self.history = [(np.copy(self.weights), self.bias)]
        self.loss_history = []
        
        for epoch in range(max_epochs):
            misclassified = 0
            for i in range(len(self.X)):
                prediction = self.predict(self.X[i])
                if prediction != self.y[i]:
                    self.weights += self.learning_rate * self.y[i] * self.X[i]
                    self.bias += self.learning_rate * self.y[i]
                    misclassified += 1
            
            self.history.append((np.copy(self.weights), self.bias))
            self.loss_history.append(misclassified)
            
            if misclassified == 0:
                break
        
        return self.history
    
    def animate_decision_boundary(self, plot_area):
        x_min, x_max = self.X[:, 0].min() - 1, self.X[:, 0].max() + 1
        y_min, y_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                            np.arange(y_min, y_max, 0.02))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for epoch, (weights, bias) in enumerate(self.history):
            ax.clear()
            ax.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap='bwr', edgecolors='k')
            
            Z = np.sign(np.dot(np.c_[xx.ravel(), yy.ravel()], weights) + bias)
            Z = Z.reshape(xx.shape)
            ax.contour(xx, yy, Z, colors='g', levels=[0], alpha=0.5, linestyles='--')
            
            ax.set_title(f'Perceptron Training - Epoch {epoch}')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            
            plot_area.pyplot(fig)
            
    def plot_loss_curve(self, loss_area):
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(1, len(self.loss_history) + 1), self.loss_history, marker='o', linestyle='-')
        ax.set_title('Loss Curve (Misclassified Points per Epoch)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Misclassified Points')
        ax.grid()
        loss_area.pyplot(fig)

def create_streamlit_app():
    st.title('Perceptron Learning Algorithm (PLA) Visualization')
    
    st.sidebar.title('PLA Theory')
    st.sidebar.markdown('''
    ### Perceptron Learning Algorithm
    
    **Key Concepts:**
    - Linear binary classifier
    - Updates weights when misclassification occurs
    - Iteratively learns a separating hyperplane
    
    **Update Rule:**
    - If misclassified: 

      w = w + η * y * x
      
      bias = bias + η * y
    
    **Limitations:**
    - Only works for linearly separable data
    - Convergence not guaranteed
    ''')
    
    tab1, tab2, tab3 = st.tabs(['Linearly Separable', 'Non-Linear', 'Noisy'])
    
    with tab1:
        st.header('Linearly Separable Dataset')
        
        X, y = make_classification(n_samples=100, n_features=2, n_classes=2,
                                   n_clusters_per_class=1, n_redundant=0, 
                                   n_informative=2, class_sep=2.0, random_state=42)
        y = 2 * y - 1
        
        max_epochs = st.slider('Max Epochs (Linearly Separable)', 1, 100, 50, 9)
        plot_area = st.empty()
        loss_area = st.empty()
        
        if st.button('Train Perceptron (Linearly Separable)'):
            pla = PerceptronVisualizer(X, y)
            pla.train(max_epochs)
            pla.animate_decision_boundary(plot_area)
            pla.plot_loss_curve(loss_area)
            st.write(f"Total Epochs: {len(pla.history) - 1}")
    
    with tab2:
        st.header('Non-Linear Dataset')
        
        X, y = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=42)
        y = 2 * y - 1
        
        max_epochs = st.slider('Max Epochs (Non-Linear)', 1, 200, 100, 19)
        plot_area = st.empty()
        loss_area = st.empty()
        
        if st.button('Train Perceptron (Non-Linear)'):
            pla = PerceptronVisualizer(X, y)
            pla.train(max_epochs)
            pla.animate_decision_boundary(plot_area)
            pla.plot_loss_curve(loss_area)
            st.write(f"Total Epochs: {len(pla.history) - 1}")
    
    with tab3:
        st.header('Noisy Dataset')
        
        X, y = make_classification(n_samples=100, n_features=2, n_classes=2,
                                   n_clusters_per_class=1, n_redundant=0,
                                   n_informative=2, class_sep=1.0,
                                   flip_y=0.1, random_state=42)
        y = 2 * y - 1
        
        max_epochs = st.slider('Max Epochs (Noisy)', 1, 200, 100, 19)
        plot_area = st.empty()
        loss_area = st.empty()
        
        if st.button('Train Perceptron (Noisy)'):
            pla = PerceptronVisualizer(X, y)
            pla.train(max_epochs)
            pla.animate_decision_boundary(plot_area)
            pla.plot_loss_curve(loss_area)
            st.write(f"Total Epochs: {len(pla.history) - 1}")

if __name__ == '__main__':
    create_streamlit_app()
