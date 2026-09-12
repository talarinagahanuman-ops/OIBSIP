# 📧 Email Spam Detection using NLP & Machine Learning

## 📌 Project Overview

Email and SMS spam messages are a common problem that can contain unwanted advertisements, fraudulent offers, phishing links, and potentially malicious content.

This project develops a **Binary Text Classification System** using **Natural Language Processing (NLP)** and **Machine Learning** to automatically classify messages into two categories:

- **Ham** – Legitimate message
- **Spam** – Unwanted or potentially fraudulent message

The project follows a complete machine learning workflow, starting from dataset exploration and text preprocessing to feature extraction, model training, evaluation, visualization, and prediction on new messages.

---

## 🎯 Project Objectives

The main objectives of this project are:

- Understand and explore the spam message dataset
- Analyze the distribution of Spam and Ham messages
- Clean and preprocess text data
- Remove unnecessary punctuation, numbers, and stopwords
- Convert text into numerical features using TF-IDF
- Split the dataset into training and testing sets
- Train a Multinomial Naive Bayes classifier
- Train a Logistic Regression classifier
- Compare the performance of both models
- Evaluate models using multiple classification metrics
- Analyze model performance using a confusion matrix
- Understand the importance of recall in spam detection
- Visualize common words using WordClouds
- Test the trained model on new unseen messages

---

## 📊 Dataset

This project uses the **SMS Spam Collection Dataset**.

The dataset contains labeled messages belonging to two classes:

| Label | Description |
|---|---|
| `ham` | Legitimate message |
| `spam` | Spam message |

### Dataset Characteristics

- **Total messages:** 5,572
- **Task:** Binary Text Classification
- **Classes:** Ham and Spam
- **Text column:** Message
- **Target column:** Label

The original dataset contains additional unused columns, which were removed during data cleaning.

---

## 🔍 Exploratory Data Analysis

The dataset was explored to understand its structure and class distribution.

The following steps were performed:

- Displayed the first few records
- Checked dataset information
- Checked missing values
- Removed unnecessary columns
- Renamed columns for better readability
- Removed duplicate messages
- Analyzed the number of Spam and Ham messages
- Calculated class percentages
- Created a bar chart showing class distribution

Understanding the class distribution is important because the dataset contains significantly more legitimate messages than spam messages.

---

## 🧹 Text Preprocessing

Raw text cannot be directly provided to most machine learning algorithms.

Therefore, the messages were cleaned using Natural Language Processing techniques.

The following preprocessing steps were applied:

1. Converted all text to lowercase
2. Removed punctuation
3. Removed numerical characters
4. Removed English stopwords
5. Removed unnecessary whitespace

The original message was preserved, while the processed text was stored in a separate column called:

```text
clean_message
````

### Example

**Original:**

```text
Congratulations! You have won $1000. Call now!
```

**After preprocessing:**

```text
congratulations won call
```

This cleaned text was then used for feature extraction.

---

## 🔢 TF-IDF Feature Extraction

### What is TF-IDF?

**TF-IDF** stands for:

> Term Frequency – Inverse Document Frequency

Machine learning algorithms require numerical input, so text messages need to be converted into numerical features.

TF-IDF assigns importance to words based on how frequently they occur in a message and how common they are across the entire dataset.

### TF – Term Frequency

Measures how frequently a word occurs in a particular message.

### IDF – Inverse Document Frequency

Reduces the importance of words that appear in many messages and gives more importance to words that are more specific.

### TF-IDF

TF-IDF combines both values to produce a numerical representation of the text.

In this project, TF-IDF was fitted only on the training data and then used to transform the testing data.

This prevents **data leakage** from the test set into the training process.

---

## ✂️ Train/Test Split

The dataset was divided into:

* **80% Training Data**
* **20% Testing Data**

The training dataset was used to train the machine learning models.

The testing dataset was used to evaluate how well the models perform on unseen messages.

A **stratified split** was used so that the proportion of Spam and Ham messages remained similar in both training and testing datasets.

---

# 🤖 Machine Learning Models

Two machine learning algorithms were implemented.

## 1. Multinomial Naive Bayes

Multinomial Naive Bayes is a popular algorithm for text classification.

It works particularly well with numerical text representations such as word counts and TF-IDF features.

In this project, Multinomial Naive Bayes was trained using the TF-IDF features to classify messages as Spam or Ham.

---

## 2. Logistic Regression

Logistic Regression was implemented as an alternative classification algorithm.

It is commonly used for binary classification problems and can effectively work with high-dimensional TF-IDF text features.

The performance of Logistic Regression was compared with Multinomial Naive Bayes.

---

# 📈 Model Evaluation

The models were evaluated using the following metrics:

### Accuracy

Measures the percentage of total predictions that were correct.

### Precision

Measures how many messages predicted as Spam were actually Spam.

### Recall

Measures how many actual Spam messages were successfully detected.

### F1 Score

The F1 Score is the harmonic mean of Precision and Recall.

It provides a balanced measure when both Precision and Recall are important.

---

## 🏆 Model Performance Comparison

| Model                       |   Accuracy |  Precision |     Recall |   F1 Score |
| --------------------------- | ---------: | ---------: | ---------: | ---------: |
| **Multinomial Naive Bayes** | **96.62%** | **98.98%** | **74.05%** | **84.72%** |
| Logistic Regression         |     95.36% |     98.82% |     64.21% |     77.78% |

### Best Performing Model

Based on the test results, **Multinomial Naive Bayes** performed better overall.

It achieved:

* Higher Accuracy
* Higher Precision
* Higher Recall
* Higher F1 Score

compared with Logistic Regression.

Therefore, Multinomial Naive Bayes was selected as the preferred model for this project.

---

# 📊 Model Performance Visualization

A grouped bar chart was created to compare:

* Accuracy
* Precision
* Recall
* F1 Score

for both machine learning models.

This visualization makes it easier to understand the difference in model performance.

---

# 🔲 Confusion Matrix

A confusion matrix was created for the Multinomial Naive Bayes model.

The confusion matrix contains four possible outcomes:

| Actual | Predicted | Meaning        |
| ------ | --------- | -------------- |
| Ham    | Ham       | True Negative  |
| Ham    | Spam      | False Positive |
| Spam   | Spam      | True Positive  |
| Spam   | Ham       | False Negative |

### Why are False Negatives Important?

A **False Negative** occurs when an actual Spam message is incorrectly classified as Ham.

This is important because a spam detection system should minimize the number of spam messages that are missed.

---

# 🎯 Why Recall is Important in Spam Detection

Recall is particularly important in spam detection because it measures how many of the actual spam messages were successfully identified.

For this project:

**Multinomial Naive Bayes Recall: 74.05%**

**Logistic Regression Recall: 64.21%**

Therefore, Multinomial Naive Bayes detected a larger proportion of the actual spam messages.

A low recall means that some spam messages are incorrectly classified as legitimate messages.

However, recall should not be considered alone.

Very aggressive spam detection could increase false positives, causing legitimate messages to be incorrectly classified as spam.

Therefore, a practical spam detection system should aim for a good balance between:

* Precision
* Recall
* F1 Score

---

# ☁️ WordCloud Analysis

As an additional visualization, WordClouds were created separately for:

* Spam messages
* Ham messages

The WordClouds help identify frequently occurring words and language patterns within each category.

This provides a simple visual understanding of the differences between Spam and Ham messages.

---

# 🧪 New Message Prediction

The trained Multinomial Naive Bayes model can also classify new, unseen messages.

The prediction process follows the same pipeline:

```text
New Message
     ↓
Text Preprocessing
     ↓
TF-IDF Transformation
     ↓
Naive Bayes Model
     ↓
Spam / Ham Prediction
```

### Example Spam Message

```text
Congratulations! You have won a free prize. Click now to claim your reward.
```

Expected prediction:

```text
Spam
```

### Example Ham Message

```text
Hey, are we meeting for lunch today?
```

Expected prediction:

```text
Ham
```

---

# 🛠️ Technologies Used

| Technology       | Purpose                        |
| ---------------- | ------------------------------ |
| Python           | Programming language           |
| Pandas           | Data manipulation and analysis |
| NumPy            | Numerical operations           |
| NLTK             | Natural Language Processing    |
| Scikit-learn     | Machine Learning               |
| Matplotlib       | Data visualization             |
| Seaborn          | Statistical visualization      |
| WordCloud        | Text visualization             |
| Jupyter Notebook | Development environment        |
| VS Code          | Project development            |

---

# 📁 Project Structure

```text
Email-Spam-Detection/
│
├── data/
│   └── spam.csv
│
├── notebooks/
│   └── spam_detection.ipynb
│
├── README.md
│
└── requirements.txt
```

---

# ⚙️ Installation & Setup

## 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

## 2. Navigate to the Project

```bash
cd Email-Spam-Detection
```

## 3. Install Required Libraries

```bash
pip install -r requirements.txt
```

## 4. Open the Notebook

Open:

```text
notebooks/spam_detection.ipynb
```

using Jupyter Notebook or VS Code.

## 5. Run the Notebook

Run the notebook cells sequentially from beginning to end.

---

# 📦 Requirements

The project uses the following Python libraries:

```text
numpy
pandas
scikit-learn
nltk
matplotlib
seaborn
wordcloud
jupyter
```

The complete list is available in:

```text
requirements.txt
```

---

# 📌 Key Learnings

Through this project, the following concepts were implemented and practiced:

* Data cleaning
* Exploratory Data Analysis
* Class distribution analysis
* Natural Language Processing
* Text preprocessing
* Stopword removal
* TF-IDF feature extraction
* Train/test splitting
* Stratified sampling
* Multinomial Naive Bayes
* Logistic Regression
* Binary classification
* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* WordCloud visualization
* New text prediction
* Model comparison

---

# 🚀 Future Improvements

The project can be further improved by:

* Hyperparameter tuning
* Testing additional machine learning algorithms
* Implementing stemming or lemmatization
* Trying n-gram features
* Using word embeddings
* Improving spam recall
* Adding prediction probability/confidence
* Building an interactive Streamlit application
* Deploying the model as a web application
* Creating an API for real-time spam detection

---

# 💼 Project Highlights

### Problem

Identify whether a message is Spam or legitimate.

### Solution

Built an NLP-based binary classification system using TF-IDF and machine learning.

### Models

* Multinomial Naive Bayes
* Logistic Regression

### Best Model

**Multinomial Naive Bayes**

### Best Accuracy

**96.62%**

### Best Spam Precision

**98.98%**

### Best Spam Recall

**74.05%**

### Best F1 Score

**84.72%**

---

# 👨‍💻 Author

**sai kiran**

B.Tech Computer Science Engineering Graduate

---

## ⭐ Conclusion

This project demonstrates how Natural Language Processing and Machine Learning can be combined to build a practical spam message detection system.

The final solution successfully processes raw text, converts it into numerical TF-IDF features, trains multiple classification models, evaluates their performance using several metrics, and predicts whether new messages are Spam or Ham.

**Multinomial Naive Bayes achieved the best overall performance on the test dataset with an accuracy of 96.62%.**

```

### One important thing before you save it

Your project is specifically **SMS Spam Detection**, rather than email-only detection, because the dataset contains SMS messages. For a more accurate GitHub title, I recommend:

**`SMS Spam Detection using NLP & Machine Learning`**

You can still call it **Email Spam Detection** if that's the project title your assignment requires, but **SMS Spam Detection** is technically more accurate for this dataset.
```
