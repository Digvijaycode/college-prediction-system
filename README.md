# College Prediction System 🎓

An AI-powered college prediction system using Machine Learning to help students find the best college matches based on their CET scores, location preferences, and course choices.

## ✨ Features

- **Smart Predictions**: ML model trained on 12,000+ admission records
- **Top 7 Matches**: Get your best college recommendations with match percentages
- **Clean Data**: 422 colleges, 140 courses, 164 locations
- **Easy to Use**: Beautiful Streamlit web interface

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 Data Cleaning

The system automatically cleans the dataset to remove:
- ✅ Duplicate entries
- ✅ Special characters (&, extra symbols)
- ✅ Inconsistent location names (e.g., "Raigad" vs "Raigad.")
- ✅ Institute code numbers
- ✅ Invalid/malformed entries

To re-clean and retrain the model:
```bash
python clean_data.py
python train_model.py
```

## 📁 Files

- `streamlit_app.py` - Main web application
- `clean_data.py` - Data cleaning script
- `train_model.py` - Model training script
- `dataset_with_location.csv` - Original dataset
- `dataset_cleaned.csv` - Cleaned dataset
- `model.pkl` - Trained ML model
- `label_encoders.pkl` - Feature encoders
- `target_encoder.pkl` - Target encoder
- `requirements.txt` - Python dependencies

## 🌐 Deploy Online

### Streamlit Cloud (Free)

1. Push your code to GitHub:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with GitHub
4. Click "New app"
5. Select your repository
6. Set main file: `streamlit_app.py`
7. Click "Deploy"

Your app will be live at: `https://<username>-<repo>.streamlit.app`

## 📈 Model Performance

- **Training Accuracy**: 82%
- **Testing Accuracy**: 39%
- **Dataset Size**: 12,772 records
- **Colleges**: 422 institutes
- **Courses**: 140 unique courses
- **Locations**: 164 cities/areas

## 👨‍💻 Developer

**Digvijay Hande**
- [LinkedIn](https://www.linkedin.com/in/digvijay-hande-1bb538264/)
- [GitHub](https://github.com/Digvijaycode)

## 📄 License

© 2025 College Prediction System. All rights reserved.
