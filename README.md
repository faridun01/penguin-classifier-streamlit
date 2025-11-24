# 🐧 Interactive Penguin Species Classifier



[Image of a penguin species]


This project presents a **complete, end-to-end machine learning pipeline** built as a fully interactive application using **Streamlit**. It allows users to explore the famous Palmer Penguins dataset, train and evaluate classification models directly within the app, and make real-time predictions.

---

## 📘 Project Overview

A dynamic showcase for applying machine learning concepts, data visualization with Python.

It allows users to:

* **Explore** and analyze the characteristics of the three different penguin species (Adelie, Chinstrap, and Gentoo).
* **Visualize** the relationships between various physical features.
* **Train and compare** three popular scikit-learn classification models.
* **Make real-time species predictions** based on numeric inputs.

Everything is computed **directly inside the app**, ensuring it's completely interactive and self-contained—no external training pipeline is required.

---

## 🔍 Key Features

### 📂 Dataset Exploration

* Full dataset preview.
* Summary **statistics** (mean, min, max, etc.).
* Class distributions for:
    * **Species**
    * **Island**
    * **Gender**

### 📊 Interactive Visualizations

* Default scatter plot visualizing key feature relationships.
* **Custom scatter plot** (choose any X and Y features).
* **Histograms** for numeric feature distributions.
* **Bar charts** for categorical distributions.

### 🤖 Machine Learning Models Included

The following `scikit-learn` models are trained on the fly inside the app:

1.  **K-Nearest Neighbors (KNN)**
2.  **Decision Tree Classifier**
3.  **Random Forest Classifier**

### 📈 Model Evaluation Tools

* **Accuracy comparison table** to easily rank model performance.
* **Confusion matrix viewer** for detailed error analysis.
* **Automatic best-model detection** based on accuracy.

### 🔮 Prediction Interface

Users can input four numeric features using interactive sliders:

1.  **Bill Length** (mm)
2.  **Bill Depth** (mm)
3.  **Flipper Length** (mm)
4.  **Body Mass** (g)

The app instantly outputs:

* **Predicted species** (e.g., Adelie, Gentoo).
* **Class probabilities** for each species.
* **Binary class indicators** (0 or 1) for easy interpretation.

---

## 🛠️ Technologies Used

| Technology | Purpose |
| :--- | :--- |
| **Python** 3.13 | Core language for the application. |
| **Streamlit** | Creating the interactive web application interface. |
| **Pandas** | Data manipulation and analysis. |
| **scikit-learn** | Machine learning model training and evaluation. |
| **Altair** | Declarative statistical visualizations. |
| **NumPy** | Numerical operations and array handling. |

---

## 🚀 Live Demo

▶️ **Try the Application** [(https://my-first-str-app.streamlit.app/)]

---

The application will open automatically in your web browser.
