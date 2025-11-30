import streamlit as st
import pickle
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="College Predictor",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load model and encoders
@st.cache_resource
def load_models():
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('label_encoders.pkl', 'rb') as enc_file:
        label_encoders = pickle.load(enc_file)
    with open('target_encoder.pkl', 'rb') as target_file:
        le_target = pickle.load(target_file)
    return model, label_encoders, le_target

try:
    model, label_encoders, le_target = load_models()
    
    # Custom CSS for clean professional look
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        .main {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }
        
        .block-container {
            max-width: 900px;
            padding: 2rem 1rem;
        }
        
        h1 {
            color: #2c3e50;
            font-weight: 700;
            text-align: center;
            font-size: 2.5rem !important;
            margin-bottom: 0.5rem !important;
        }
        
        h2 {
            color: #34495e;
            font-weight: 600;
        }
        
        h3 {
            color: #34495e;
            font-weight: 600;
        }
        
        .stButton>button {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white !important;
            font-size: 1.1rem;
            font-weight: 600;
            padding: 0.8rem 2rem;
            border-radius: 8px;
            border: none;
            width: 100%;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
        }
        
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
        }
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: white;
            padding: 1rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 0 24px;
            font-weight: 600;
            font-size: 0.95rem;
            color: #555;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white !important;
        }
        
        .stSelectbox label, .stNumberInput label {
            font-weight: 600 !important;
            font-size: 0.95rem !important;
            color: #2c3e50 !important;
            margin-bottom: 0.5rem !important;
        }
        
        .stSelectbox > div > div, .stNumberInput > div > div > input {
            background-color: white !important;
            border: 2px solid #e1e8ed !important;
            border-radius: 8px !important;
            font-size: 1rem !important;
            font-weight: 500 !important;
            color: #2c3e50 !important;
        }
        
        .stSelectbox > div > div:hover, .stNumberInput > div > div > input:hover {
            border-color: #667eea !important;
        }
        
        .stSelectbox > div > div:focus-within, .stNumberInput > div > div > input:focus {
            border-color: #667eea !important;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        }
        
        div[data-baseweb="select"] > div {
            background-color: white !important;
            font-size: 1rem !important;
        }
        
        .stInfo {
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
            border-radius: 8px;
            padding: 1rem;
        }
        
        .stSuccess {
            background-color: #e8f5e9;
            border-left: 4px solid #4caf50;
            border-radius: 8px;
            padding: 1rem;
        }
        
        .stError {
            background-color: #ffebee;
            border-left: 4px solid #f44336;
            border-radius: 8px;
            padding: 1rem;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display: none;}
        </style>
    """, unsafe_allow_html=True)

    # Main header
    st.markdown("""
        <div style='text-align: center; padding: 2rem 1rem; background: white; border-radius: 12px; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 2rem;'>
            <h1 style='color: #2c3e50; margin: 0;'>🎓 College Prediction System</h1>
            <p style='color: #7f8c8d; font-size: 1.1rem; margin: 0.5rem 0 0 0;'>
                Find Your Perfect College Match Using AI & Machine Learning
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🏠 Home", "🔮 Predict", "📚 Resources"])

    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("""
                <div style='background: white; padding: 1.8rem; border-radius: 12px; 
                box-shadow: 0 2px 8px rgba(0,0,0,0.08); height: 100%;'>
                    <h3 style='color: #667eea; margin-bottom: 1rem;'>🎯 What We Offer</h3>
                    <ul style='line-height: 2.2; color: #2c3e50; list-style-position: inside;'>
                        <li>AI-Powered predictions</li>
                        <li>Top 7 college recommendations</li>
                        <li>Match percentage analysis</li>
                        <li>CET score based filtering</li>
                        <li>Location preferences</li>
                        <li>Course-specific results</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div style='background: white; padding: 1.8rem; border-radius: 12px; 
                box-shadow: 0 2px 8px rgba(0,0,0,0.08); height: 100%;'>
                    <h3 style='color: #764ba2; margin-bottom: 1rem;'>📖 How to Use</h3>
                    <ol style='line-height: 2.2; color: #2c3e50; padding-left: 1.2rem;'>
                        <li>Go to <strong>Predict</strong> tab</li>
                        <li>Select your location</li>
                        <li>Enter CET score (0-100)</li>
                        <li>Choose your course</li>
                        <li>Click <strong>Predict</strong></li>
                        <li>View your results</li>
                    </ol>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 **Tip:** Our ML model analyzes historical admission data to provide accurate predictions!")
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; color: #7f8c8d; padding: 1rem;'>
                <p style='margin: 5px;'><strong>Developed by 
                <a href='https://www.linkedin.com/in/digvijay-hande-1bb538264/' target='_blank' 
                style='color: #667eea; text-decoration: none;'>Digvijay Hande</a></strong></p>
                <p style='margin: 5px;'>
                <a href='https://github.com/Digvijaycode' target='_blank' 
                style='color: #764ba2; text-decoration: none;'>GitHub</a> | © 2025
                </p>
            </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
            <div style='background: white; padding: 1.5rem; border-radius: 12px; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 2rem; text-align: center;'>
                <h2 style='color: #667eea; margin: 0;'>🔮 Predict Your College</h2>
                <p style='color: #7f8c8d; margin: 0.5rem 0 0 0;'>
                    Enter your details to get personalized recommendations
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Get unique values
        locations = sorted(list(label_encoders['Location'].classes_))
        courses = sorted(list(label_encoders['Course Name'].classes_))
        
        # Create form
        with st.form("prediction_form", clear_on_submit=False):
            st.markdown("""
                <div style='background: white; padding: 2rem; border-radius: 12px; 
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3, gap="medium")
            
            with col1:
                location = st.selectbox(
                    "📍 Location",
                    options=locations,
                    index=0
                )
            
            with col2:
                cet_score = st.number_input(
                    "📊 CET Score",
                    min_value=0.0,
                    max_value=100.0,
                    value=50.0,
                    step=0.5,
                    format="%.1f"
                )
            
            with col3:
                course = st.selectbox(
                    "📚 Course",
                    options=courses,
                    index=0
                )
            
            st.markdown("</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            submit = st.form_submit_button("🔍 Predict Colleges")
        
        if submit:
            with st.spinner("🔄 Analyzing your profile..."):
                try:
                    # Prepare and encode input
                    input_data = [[cet_score, course, location]]
                    input_data[0][1] = label_encoders['Course Name'].transform([input_data[0][1]])[0]
                    input_data[0][2] = label_encoders['Location'].transform([input_data[0][2]])[0]
                    
                    # Get predictions
                    predictions = model.predict_proba(input_data)[0]
                    top_indices = predictions.argsort()[-7:][::-1]
                    top_predictions = [model.classes_[i] for i in top_indices]
                    institutes = le_target.inverse_transform(top_predictions).tolist()
                    probabilities = [predictions[i] * 100 for i in top_indices]
                    
                    st.success("✅ Prediction Complete!")
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    st.markdown("<h3 style='text-align: center; color: #667eea;'>🏆 Your Top 7 College Matches</h3>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    for idx, (institute, prob) in enumerate(zip(institutes, probabilities), 1):
                        if idx == 1:
                            badge = "🥇"
                            color = "#FFD700"
                        elif idx == 2:
                            badge = "🥈"
                            color = "#C0C0C0"
                        elif idx == 3:
                            badge = "🥉"
                            color = "#CD7F32"
                        else:
                            badge = f"{idx}"
                            color = "#667eea"
                        
                        st.markdown(f"""
                            <div style='background: white; padding: 1.5rem; border-radius: 12px; 
                            margin: 1rem 0; box-shadow: 0 2px 8px rgba(0,0,0,0.08); 
                            border-left: 4px solid {color};'>
                                <div style='display: flex; justify-content: space-between; align-items: center;'>
                                    <div style='display: flex; align-items: center; gap: 1rem; flex: 1;'>
                                        <span style='font-size: 1.8rem; font-weight: bold;'>{badge}</span>
                                        <div style='flex: 1;'>
                                            <h4 style='margin: 0; color: #2c3e50; font-size: 1.1rem;'>{institute}</h4>
                                            <p style='margin: 0.3rem 0 0 0; color: #7f8c8d; font-size: 0.9rem;'>
                                                {location} • {course}
                                            </p>
                                        </div>
                                    </div>
                                    <div style='background: {color}; color: white; padding: 0.6rem 1.2rem; 
                                    border-radius: 20px; font-weight: 600; font-size: 0.95rem;'>
                                        {prob:.1f}%
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.info("💡 Match percentages indicate admission probability based on historical data")
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("""
                <div style='background: white; padding: 1.8rem; border-radius: 12px; 
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
                    <h3 style='color: #667eea; margin-bottom: 1rem;'>🔗 Official Portals</h3>
                    <ul style='line-height: 2.5; color: #2c3e50; list-style: none; padding: 0;'>
                        <li>🌐 <a href='https://cetcell.mahacet.org' target='_blank' 
                        style='color: #667eea; text-decoration: none; font-weight: 500;'>
                        CET Cell Website</a></li>
                        <li>🌐 <a href='https://jeemain.nta.nic.in' target='_blank' 
                        style='color: #667eea; text-decoration: none; font-weight: 500;'>
                        JEE Mains Portal</a></li>
                        <li>🌐 <a href='https://neet.nta.nic.in' target='_blank' 
                        style='color: #667eea; text-decoration: none; font-weight: 500;'>
                        NEET Portal</a></li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div style='background: white; padding: 1.8rem; border-radius: 12px; 
                box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
                    <h3 style='color: #764ba2; margin-bottom: 1rem;'>ℹ️ About</h3>
                    <p style='line-height: 1.8; color: #2c3e50;'>
                        This system uses <strong>Machine Learning</strong> trained on historical 
                        admission data to predict college matches.
                    </p>
                    <p style='line-height: 1.8; color: #2c3e50; margin-top: 1rem;'>
                        <strong>Analysis based on:</strong><br>
                        • CET scores<br>
                        • Location preferences<br>
                        • Course requirements<br>
                        • Historical trends
                    </p>
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        st.markdown("<h3 style='text-align: center; color: #667eea;'>📊 System Stats</h3>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        stats = [
            ("🎓", "358", "Colleges", "#667eea"),
            ("📚", "108", "Courses", "#764ba2"),
            ("📍", "124", "Locations", "#2196f3")
        ]
        
        for col, (icon, value, label, color) in zip([col1, col2, col3], stats):
            with col:
                st.markdown(f"""
                    <div style='background: white; padding: 1.5rem; border-radius: 12px; 
                    text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08);'>
                        <div style='font-size: 2rem;'>{icon}</div>
                        <div style='color: {color}; font-size: 1.8rem; font-weight: 700; margin: 0.5rem 0;'>{value}</div>
                        <div style='color: #7f8c8d; font-size: 0.9rem;'>{label}</div>
                    </div>
                """, unsafe_allow_html=True)

except FileNotFoundError:
    st.error("⚠️ **Model files not found!**")
    st.info("Please ensure model.pkl, label_encoders.pkl, and target_encoder.pkl exist in the directory.")
except Exception as e:
    st.error(f"❌ **Error:** {str(e)}")
