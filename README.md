# 🤖 Project 2: Data Classification Using AI (Iris Dataset & KNN)

Welcome to Project 2 of the Artificial Intelligence Industrial Training Kit powered by DecodeLabs! This project focuses on **Supervised Learning** by training a machine learning model to recognize patterns and categorize data.

## 🌟 Project Overview
This project implements a **K-Nearest Neighbors (KNN)** classifier to categorize Iris flowers into three species (*Setosa*, *Versicolor*, and *Virginica*) using four physical dimensions (sepal and petal length/width).

## ⚙️ Pipeline Architecture
1. **📥 Input & Loading**: Loaded 150 balanced samples from the Iris benchmark dataset.
2. **🔀 Structural Split**: Divided data into an **80% training set** (for pattern recognition) and a **20% test set** (for validation) with random shuffling[cite: 1].
3. **⚖️ Feature Scaling**: Standardized features using `StandardScaler` ($\text{Mean} = 0, \text{Variance} = 1$) to prevent feature bias[cite: 1].
4. **⚙️ Model Training**: Applied the **KNN algorithm** based on the proximity principle[cite: 1].
5. **📊 Output Validation**: Evaluated performance using accuracy metrics, a **Confusion Matrix**, and a classification report[cite: 1].

## 💻 How to Run
1. Ensure Python is installed along with the required libraries[cite: 1]:
   ```bash
   pip install scikit-learn numpy scipy