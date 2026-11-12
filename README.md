# 🎓 Student Performance Predictor

An AI-based student performance prediction system built using Machine Learning and Artificial Neural Networks (ANN). The system analyzes academic performance, attendance, study habits, and lifestyle-related factors to classify students into **High, Medium, or Low** performance categories.

## 🚀 Features

- Predict student performance using an Artificial Neural Network.
- Compare four Machine Learning models.
- Interactive web interface built with Streamlit.
- Flask REST API for model predictions.
- Prediction confidence and class probability visualization.
- Support for multiple student input features.
- Compare All Models functionality.

## 🧠 Machine Learning Models

The project implements and compares the following models:

1. Logistic Regression
2. Random Forest
3. Support Vector Machine (SVM)
4. Artificial Neural Network (ANN)

### ANN Architecture

The Artificial Neural Network uses the following architecture:

- Input Layer: 10 features
- Hidden Layer 1: 64 neurons with ReLU activation
- Dropout: 0.2
- Hidden Layer 2: 32 neurons with ReLU activation
- Dropout: 0.2
- Output Layer: 3 neurons with Softmax activation

**Training configuration:**
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Batch Size: 32
- Maximum Epochs: 15
- Early Stopping: Enabled

## 📊 Dataset

The project uses a dataset containing **15,000 student records** with 10 input features.

### Input Features

1. Age
2. Attendance Percentage
3. Study Hours Per Week
4. Previous Score
5. Assignment Completion Percentage
6. Internal Assessment Score
7. Participation Score
8. Sleep Hours Per Day
9. Extracurricular Hours Per Week
10. Failed Subjects

### Target Classes

- High
- Medium
- Low

## 📈 Model Performance

The models were evaluated using Accuracy, Precision, Recall, and F1-score.

| Model | Accuracy |
|---|---:|
| Logistic Regression | 69.20% |
| Random Forest | 66.03% |
| Support Vector Machine | 68.17% |
| ANN | 68.57% |

**Observation:** Logistic Regression achieved the highest accuracy in the reported comparison, while ANN provided competitive performance.

## 🛠️ Technologies Used

- Python
- TensorFlow and Keras
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Flask
- Joblib

## 📁 Project Structure

```text
student-performance-predictor/
│
├── app.py
├── api.py
├── requirements.txt
├── README.md
│
├── student_performance_ann.keras
├── Logistic_Regression.pkl
├── Random_Forest.pkl
├── Support_Vector_Machine.pkl
├── scaler.pkl
└── label_encoder.pkl
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/sunnyrai2006/student-performance-predictor.git
```

```bash
cd student-performance-predictor
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the environment

For Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the Flask API

Open the first terminal:

```bash
python api.py
```

The API will run at:

```text
http://127.0.0.1:5000/
```

### 6. Start the Streamlit application

Open a second terminal:

```bash
streamlit run app.py
```

The application will open in your browser.

## 🔗 API Endpoint

**Prediction Endpoint:**

```text
POST http://127.0.0.1:5000/predict
```

The API accepts student information in JSON format and returns the predicted performance category, confidence where available, and class probabilities.

## 📋 Evaluation Metrics

The project uses the following evaluation methods:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Model Comparison

## 🔮 Future Improvements

- Improve prediction accuracy through hyperparameter tuning.
- Explore additional student-related features.
- Improve classification performance for the Medium category.
- Add interactive performance analytics and visualizations.
- Explore deployment on a cloud platform.

## ⚠️ Disclaimer

This project is developed for educational and academic purposes. Predictions are based on the trained dataset and should not be considered a definitive assessment of a student's academic ability or future performance.

## 👨‍💻 Project Information

**Project Name:** AI-Based Student Performance Predictor

**Domain:** Artificial Intelligence and Machine Learning

**Application:** Student Performance Classification

**Developer:** Sunny Rai
