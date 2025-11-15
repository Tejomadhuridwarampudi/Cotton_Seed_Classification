# Cotton Seeds Classification – Streamlit App

This project is a **Machine Learning Web Application** built using **Streamlit**.
It allows users to upload a cotton seed dataset, clean the data, scale features, and run ML models like:

- **Novel Tangent Decision Tree (NTGT Model)**
- **Random K-Value KNN Model (RKN Model)**

The app also displays confusion matrices and accuracy scores for better understanding of model performance.

## 🚀 Features

### 1. Upload Dataset
- Upload your cotton seed CSV file
- Preview your dataset directly in the app

### 2. Data Cleaning
- Handle missing values using mean imputation
- Feature scaling using MinMaxScaler

### 3. Models Included
- Decision Tree (Entropy)
- K-Nearest Neighbors (k=3)

### 4. Evaluation Metrics
- Confusion Matrix
- Heatmap visualizations
- Accuracy Score

## 📂 Project Structure

```
├── bean.py
├── requirements.txt
├── CS.jpg
├── MARKS.jpg
├── 123.png
└── .streamlit/
    └── config.toml
```

## ▶️ How to Run Locally

1. Clone the repository:
```
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run the Streamlit app:
```
streamlit run bean.py
```

## 🌐 Deployment (Streamlit Cloud)

1. Push all files to GitHub  
2. Go to **https://share.streamlit.io**  
3. Click **Deploy App**  
4. Select your repository and choose `bean.py`  
5. Your app will go live with a public link.

## 👩‍💻 Author

Your Name  
Roll Number (optional)  
Cotton Seed Classification Project – B.Tech 2nd Year
