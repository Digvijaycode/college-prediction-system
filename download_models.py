import os
import pickle
import gdown
import streamlit as st

# Google Drive file IDs for model files
MODEL_FILES = {
    'model.pkl': '1YOUR_DRIVE_ID_MODEL',
    'label_encoders.pkl': '1YOUR_DRIVE_ID_ENCODERS',
    'target_encoder.pkl': '1YOUR_DRIVE_ID_TARGET'
}

def download_model_files():
    """Download model files from Google Drive"""
    for filename, drive_id in MODEL_FILES.items():
        if not os.path.exists(filename):
            st.info(f"📥 Downloading {filename}...")
            try:
                gdown.download(f'https://drive.google.com/uc?id={drive_id}', filename, quiet=False)
                st.success(f"✅ Downloaded {filename}")
            except Exception as e:
                st.error(f"❌ Failed to download {filename}: {str(e)}")
                return False
    return True

if __name__ == "__main__":
    if not all(os.path.exists(f) for f in MODEL_FILES.keys()):
        if not download_model_files():
            st.stop()
