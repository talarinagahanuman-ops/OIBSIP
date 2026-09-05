# Iris Flower Classification

## OIBSIP Data Science Task 1

### Project Overview

This project uses Machine Learning to classify Iris flowers into three different species:

* Setosa
* Versicolor
* Virginica

The classification is performed using measurements of the flower's sepal and petal.

### Dataset

The Iris dataset is loaded directly from the Scikit-learn library using `load_iris()`.

The dataset contains:

* 150 flower samples
* 4 numerical features
* 3 Iris species

### Features Used

The model uses the following four features:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

### Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

### Exploratory Data Analysis

The following analysis was performed:

* Dataset shape
* Data types
* Missing-value checking
* Descriptive statistics
* Species distribution
* Pairplot
* Boxplot
* Correlation heatmap

### Machine Learning Models

Two classification algorithms were trained:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)

### Model Evaluation

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

The models were compared to identify the better-performing classifier.

### New Flower Prediction

A new Iris flower was provided to the trained KNN model using the following measurements:

* Sepal Length: 5.1 cm
* Sepal Width: 3.5 cm
* Petal Length: 1.4 cm
* Petal Width: 0.2 cm

The model predicted:

**Setosa**

### Project Structure

```text
DataScience-Task1-IrisClassification/
│
├── iris_classification.ipynb
├── requirements.txt
├── README.md
│
└── images/
    ├── pairplot.png
    ├── boxplot.png
    ├── species_count.png
    ├── correlation_heatmap.png
    ├── model_accuracy.png
    ├── logistic_regression_confusion_matrix.png
    └── knn_confusion_matrix.png
```

### How to Run the Project

#### 1. Create a virtual environment

```bash
python -m venv venv
```

#### 2. Activate the virtual environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

#### 3. Install the required libraries

```bash
pip install -r requirements.txt
```

#### 4. Open the Jupyter Notebook

Open:

```text
iris_classification.ipynb
```

in VS Code.

#### 5. Run the notebook

Run the cells from top to bottom.

### Conclusion

This project demonstrates how Machine Learning can be used to classify Iris flower species based on their physical measurements.

Logistic Regression and K-Nearest Neighbors were trained and evaluated using multiple performance metrics. The project also demonstrates how a trained model can predict the species of a new flower.
