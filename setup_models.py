#!/usr/bin/env python3
"""
Setup script to prepare model files for Streamlit Cloud deployment.
This creates placeholder files and instructions for uploading real models.
"""

import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

def create_dummy_models():
    """Create minimal dummy model files for testing deployment"""
    print("Creating dummy model files for deployment...")
    
    # Create dummy RandomForest model
    dummy_model = RandomForestClassifier(n_estimators=10, random_state=42)
    X_dummy = np.array([[50, 1, 1], [75, 2, 2]])
    y_dummy = np.array([0, 1])
    dummy_model.fit(X_dummy, y_dummy)
    
    # Create label encoders
    label_encoders = {}
    for col_name in ['Course Name', 'Location']:
        le = LabelEncoder()
        le.fit(['Option1', 'Option2', 'Option3'])
        label_encoders[col_name] = le
    
    # Create target encoder
    le_target = LabelEncoder()
    le_target.fit([f'College{i}' for i in range(358)])
    
    # Save files
    with open('model.pkl', 'wb') as f:
        pickle.dump(dummy_model, f)
    print("✅ model.pkl created")
    
    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    print("✅ label_encoders.pkl created")
    
    with open('target_encoder.pkl', 'wb') as f:
        pickle.dump(le_target, f)
    print("✅ target_encoder.pkl created")
    
    print("\n⚠️  Note: These are dummy files for testing deployment.")
    print("To use real models, replace them with your trained models.")

if __name__ == '__main__':
    create_dummy_models()
