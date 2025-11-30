import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle

print("="*60)
print("COLLEGE PREDICTION MODEL TRAINING")
print("="*60)

# Load the cleaned dataset
print("\n1. Loading cleaned dataset...")
data = pd.read_csv('dataset_cleaned.csv')
print(f"   Dataset loaded: {len(data)} records")
print(f"   Unique Institutes: {data['Institute Name'].nunique()}")
print(f"   Unique Courses: {data['Course Name'].nunique()}")
print(f"   Unique Locations: {data['Location'].nunique()}")

# Label encode categorical features
print("\n2. Encoding categorical features...")
label_encoders = {}

for column in ['Course Name', 'Location']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le
    print(f"   ✓ {column}: {len(le.classes_)} unique values")

# Encode target variable
print("\n3. Encoding target variable (Institute Name)...")
le_target = LabelEncoder()
data['Institute Name'] = le_target.fit_transform(data['Institute Name'])
print(f"   ✓ Institute Name: {len(le_target.classes_)} unique colleges")

# Filter out institutes with very few samples (need at least 2 for stratified split)
print("\n3.5. Filtering institutes with insufficient data...")
institute_counts = data['Institute Name'].value_counts()
valid_institutes = institute_counts[institute_counts >= 2].index
data = data[data['Institute Name'].isin(valid_institutes)]
print(f"   ✓ Kept {len(valid_institutes)} institutes with 2+ samples")
print(f"   ✓ Total records: {len(data)}")

# Define features and target
features = ['Percentile', 'Course Name', 'Location']
target = 'Institute Name'

X = data[features]
y = data[target]

# Train-test split
print("\n4. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Training set: {len(X_train)} samples")
print(f"   Testing set: {len(X_test)} samples")

# Train the model with optimized parameters for better generalization
print("\n5. Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=30,
    min_samples_split=5,
    min_samples_leaf=3,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=1
)
model.fit(X_train, y_train)
print("   ✓ Model training complete!")

# Evaluate the model
print("\n6. Evaluating model performance...")
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_accuracy = accuracy_score(y_train, train_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)

print(f"   Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"   Testing Accuracy: {test_accuracy * 100:.2f}%")

# Save the model and encoders
print("\n7. Saving model and encoders...")
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)
print("   ✓ model.pkl saved")

with open('label_encoders.pkl', 'wb') as encoder_file:
    pickle.dump(label_encoders, encoder_file)
print("   ✓ label_encoders.pkl saved")

with open('target_encoder.pkl', 'wb') as target_file:
    pickle.dump(le_target, target_file)
print("   ✓ target_encoder.pkl saved")

print("\n" + "="*60)
print("✅ MODEL TRAINING COMPLETE!")
print("="*60)
print("\nYou can now run the Streamlit app:")
print("   streamlit run streamlit_app.py")
print("="*60)
