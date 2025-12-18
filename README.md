# 🎯 CS2 Match Winner Prediction (CT vs T)

This project predicts the winner of a Counter-Strike 2 match
(CT side or T side) using machine learning.

## 🚀 Problem Statement
Given match statistics such as economy, kills, round score, and map,
the model predicts which side is more likely to win the match.

## 📊 Dataset
- Source: (HLTV / Kaggle / Custom scraped)
- Features:
  - CT economy
  - T economy
  - Round difference
  - Map name
  - K/D ratio
  - Utility damage
  - Win streak

## 🧠 Model Used
- Logistic Regression / Random Forest / XGBoost
- Accuracy: **XX%**
- F1 Score: **XX**

## ⚙️ How to Run
```bash
pip install -r requirements.txt
python src/train.py
python src/predict.py
