# Sales Prediction Using Python

## 📌 Project Overview

This project uses **Machine Learning** to predict sales based on advertising expenditure on **TV, Radio, and Newspaper**.

The project compares different regression approaches and uses a **Random Forest Regression** model to make sales predictions.

---

## 🎯 Objectives

* Analyze advertising and sales data.
* Understand the relationship between advertising expenditure and sales.
* Train machine learning models for sales prediction.
* Evaluate model performance using MAE and RMSE.
* Identify the most important advertising factors.
* Predict sales for new advertising budgets.
* Save and reuse the trained machine learning model.

---

## 📊 Dataset

The dataset contains the following variables:

| Feature   | Description                          |
| --------- | ------------------------------------ |
| TV        | Advertising expenditure on TV        |
| Radio     | Advertising expenditure on Radio     |
| Newspaper | Advertising expenditure on Newspaper |
| Sales     | Sales generated                      |

**Target Variable:** `Sales`

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* Joblib
* VS Code

---

## 🤖 Machine Learning Models

The project uses:

1. **Linear Regression**
2. **Random Forest Regression**

The Random Forest model was used for the final prediction system.

---

## 📈 Model Performance

The Random Forest model achieved:

* **MAE:** 0.6203
* **RMSE:** 0.7687

Lower MAE and RMSE values indicate better prediction performance.

---

## ⭐ Feature Importance

The Random Forest model identified the following feature importance values:

| Feature   | Importance |
| --------- | ---------: |
| TV        |     0.6248 |
| Radio     |     0.3622 |
| Newspaper |     0.0130 |

TV advertising had the highest importance in the model.

---

## 🔮 Example Prediction

For the following advertising budget:

```text
TV = 150
Radio = 30
Newspaper = 20
```

The model predicted:

```text
Predicted Sales = 15.42
```

---

## 💾 Saved Model

The trained model is saved as:

```text
sales_prediction_model.pkl
```

The saved model can be loaded later using Joblib without retraining the model.

---

## ▶️ How to Run the Project

### 1. Clone or download the project

Open the project folder in VS Code.

### 2. Install required libraries

```bash
pip install pandas numpy matplotlib scikit-learn joblib
```

### 3. Run the Python program

```bash
python sales_prediction.py
```

### 4. Enter advertising values

The program asks for:

```text
Enter TV advertising budget:
Enter Radio advertising budget:
Enter Newspaper advertising budget:
```

It then displays the predicted sales.

---

## 📁 Project Structure

```text
Sales-Prediction/
│
├── sales_prediction.py
├── sales_prediction_model.pkl
├── dataset.csv
├── README.md
└── graph.png
```

---

## 📌 Conclusion

This project demonstrates how machine learning can be used to predict sales from advertising expenditure.

The Random Forest model provided good prediction performance, with an MAE of approximately **0.6203** and an RMSE of approximately **0.7687**. Feature importance analysis showed that TV advertising was the most influential feature among the three advertising channels.

---

## 🚀 Future Scope

* Use a larger and more diverse dataset.
* Add additional business-related features.
* Experiment with advanced machine learning algorithms.
* Build a web-based prediction application.
* Deploy the model as an online service.


