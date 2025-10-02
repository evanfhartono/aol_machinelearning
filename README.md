# 📊 YouTube Video Views Prediction (Machine Learning)

This project develops a **machine learning model** to predict the number of views of YouTube videos based on **video metadata** (title, tags, description, channel, category, etc.).  
It was built as part of an **Artificial Intelligence / Machine Learning coursework** and demonstrates skills in **data preprocessing, feature engineering, model training, evaluation, and web deployment**.

---

## 🚀 Project Overview
YouTube is one of the largest social platforms, with over **2 billion monthly active users**.  
Predicting video view counts is a complex problem, influenced by video metadata, platform algorithms, and audience behavior.  

This project aims to:
- 📈 Build a predictive model using **video metadata** (title, tags, description, channel name, category, publication date).
- 🧩 Explore **feature engineering** such as Word2Vec text embeddings and tag counts.
- 🤖 Train a regression model (**XGBoost**) to estimate video views.
- 🌐 Deploy the model in a simple **Flask-based web app**.

---

## 📂 Project Structure
```
youtube_views_prediction/
│
├── data/ # Raw dataset (YouTube Trending Video Dataset from Kaggle)
├── preprocessing/ # Data cleaning and feature engineering scripts
├── models/ # Trained ML models (XGBoost, Ridge regression, etc.)
├── app/ # Flask web app for prediction
├── notebooks/ # Jupyter notebooks for EDA and experiments
└── README.md # Documentation
```


---

## 🔑 Dataset
- **Source:** [YouTube Trending Video Dataset (Kaggle)](https://www.kaggle.com)  
- **Features Used:**
  - Title, tags, description (processed with **Word2Vec** embeddings)
  - Channel name (encoded)
  - Category ID
  - Publication date (year, month, day)
  - Tag count
  - Binary flags: comments_disabled, ratings_disabled  

- **Target:** `view_count`

---

## 🛠️ Methodology

### 1. Preprocessing
- Filtered English titles (≥90% ASCII).
- Dropped irrelevant columns (video_id, likes, dislikes, etc.).
- Extracted temporal features (year, month, day).
- Cleaned duplicates & missing values.
- Converted boolean flags to integers.
- Removed outliers using **IQR method**.

### 2. Feature Engineering
- **Word2Vec embeddings** for title, tags, description → averaged into numeric vectors.
- **Tag count** feature.
- Encoded channel title & category.

### 3. Model Training
- **XGBoost Regressor** with tuned hyperparameters:
  - `n_estimators = 100`
  - `max_depth = 5`
  - `learning_rate = 0.1`
- Training split: **70% train, 15% validation, 15% test** (time-sorted to prevent leakage).
- Target transformed with `log1p(view_count)`.

### 4. Deployment
- Built a **Flask web app** where users can input video metadata.
- Backend handles preprocessing, loads the trained model, and returns predictions.

---

## 📊 Results

**Evaluation Metrics (on log scale):**
- MSE = 1.1193  
- RMSE = 1.0579  
- MAE = 0.8278  
- R² = 0.0262  

⚠️ While the model captures some patterns, performance is limited by:
- Missing external factors (e.g., subscribers, viral trends).
- Noise in textual features.
- Training on trending videos only (less generalizable).

---

## 🌐 Application Workflow

1. **Input Metadata**  
   - Title, tags, description  
   - Channel name, category, publish date  
   - Flags (comments/ratings disabled)

2. **Preprocessing**  
   - Text → Word2Vec scores  
   - Temporal & categorical encoding  

3. **Prediction**  
   - XGBoost regression on log scale  

4. **Output**  
   - Estimated view count (inverse log transform)

---

## 📈 Future Improvements
- Add **video duration, subscriber count, thumbnail features**.
- Use **deep learning architectures** (LSTM, Transformers) for temporal & textual modeling.
- Improve preprocessing for noisy tags/description.
- Deploy with a more polished **frontend UI**.

---

## 🛠️ Tech Stack
- **Python**
- **Pandas, NumPy, Scikit-learn**
- **XGBoost**
- **Gensim (Word2Vec)**
- **Flask (Web App)**
- **HTML/CSS** (Frontend)
- **Joblib** (Model serialization)

---

## 🏃 How to Run the Project

Follow these steps to set up and run the project locally:

### 1. Clone the repository
```bash
git clone https://github.com/evanfhartono/aol_machinelearning.git
cd aol_machinelearning
```
### 2. Create and activate a virtual environment & Install dependencies
```
python -m venv venv
venv\Scripts\activate

pip install pandas numpy scikit-learn xgboost gensim flask joblib
```
### 3. Run the Flask app
```
python app/app.py
```
### 4. Open in browser
http://127.0.0.1:5000
---
## UI Preview
<img width="361" height="591" alt="chrome_NktA0RmuAl" src="https://github.com/user-attachments/assets/5b45f087-a73e-4f15-8a69-d8eb3d6a9c9f" />
<img width="363" height="588" alt="chrome_bTurJd5fG8" src="https://github.com/user-attachments/assets/eca8ba32-7b40-4dbf-be7c-5ec8b3357d0e" />

## Loaa and actual vs prediction comparison table
<img width="572" height="341" alt="chrome_qy7JluS0Pk" src="https://github.com/user-attachments/assets/f9e2a33a-e742-4ae3-aaac-d0821e7a081e" />
<img width="442" height="348" alt="chrome_iLbQrnBki8" src="https://github.com/user-attachments/assets/3350ad56-07de-4cc1-a2fc-47ca22e1b5c5" />


## 🙋 About
This repository is part of my **Machine Learning portfolio**.  
I worked on it to practice:
- End-to-end ML pipeline (data → model → web app)  
- NLP (Word2Vec text features)  
- Model evaluation and deployment  
