import pandas as pd
import re

# Load the dataset
print("Loading dataset...")
data = pd.read_csv('dataset_with_location.csv')

print(f"Original dataset size: {len(data)} rows")

# Clean Percentile column
print("\nCleaning Percentile column...")
data['Percentile'] = data['Percentile'].astype(str).str.extract(r'([\d\.]+)').astype(float)

# Clean Institute Name - extract just the college name without codes
print("Cleaning Institute Name...")
def clean_institute_name(name):
    if pd.isna(name):
        return name
    # Remove code numbers like "6006 - " or "3215 - "
    name = re.sub(r'^\d+\s*-\s*', '', str(name))
    # Remove extra whitespace
    name = ' '.join(name.split())
    # Remove special characters but keep essential ones
    name = re.sub(r'[^\w\s\-\'\,\.\(\)&]', '', name)
    return name.strip()

data['Institute Name'] = data['Institute Name'].apply(clean_institute_name)

# Clean Course Name
print("Cleaning Course Name...")
def clean_course_name(course):
    if pd.isna(course):
        return course
    course = str(course)
    # Remove extra whitespace and newlines
    course = ' '.join(course.split())
    # Fix common issues
    course = course.replace('&', 'and')
    # Remove special characters
    course = re.sub(r'[^\w\s\-\(\)]', '', course)
    return course.strip()

data['Course Name'] = data['Course Name'].apply(clean_course_name)

# Clean Location
print("Cleaning Location...")
def clean_location(location):
    if pd.isna(location):
        return location
    location = str(location)
    # Remove extra whitespace
    location = ' '.join(location.split())
    # Remove special characters
    location = re.sub(r'[^\w\s\-]', ' ', location)
    # Take only the main city (first word or most relevant part)
    location = location.strip()
    # Common location fixes
    if 'Mumbai' in location:
        return 'Mumbai'
    elif 'Pune' in location:
        return 'Pune'
    elif 'Nashik' in location:
        return 'Nashik'
    elif 'Nagpur' in location:
        return 'Nagpur'
    elif 'Thane' in location:
        return 'Thane'
    elif 'Navi Mumbai' in location:
        return 'Navi Mumbai'
    elif 'Aurangabad' in location:
        return 'Aurangabad'
    elif 'Kolhapur' in location:
        return 'Kolhapur'
    elif 'Raigad' in location:
        return 'Raigad'
    elif 'University' in location:
        return 'Pune'  # Most university locations are Pune
    else:
        # Take first meaningful word
        parts = location.split()
        return parts[0] if parts else location

data['Location'] = data['Location'].apply(clean_location)

# Remove rows with missing critical data
print("\nRemoving rows with missing data...")
before = len(data)
data = data.dropna(subset=['Percentile', 'Course Name', 'Location', 'Institute Name'])
after = len(data)
print(f"Removed {before - after} rows with missing data")

# Remove duplicates
print("\nRemoving duplicate rows...")
before = len(data)
data = data.drop_duplicates(subset=['Percentile', 'Course Name', 'Location', 'Institute Name'])
after = len(data)
print(f"Removed {before - after} duplicate rows")

# Filter out entries with very short or invalid names
print("\nFiltering invalid entries...")
before = len(data)
data = data[data['Institute Name'].str.len() > 5]
data = data[data['Course Name'].str.len() > 3]
data = data[data['Location'].str.len() > 2]
after = len(data)
print(f"Removed {before - after} invalid entries")

# Remove institutes with too few samples (need at least 5 for better learning)
print("\nRemoving institutes with insufficient data...")
before = len(data)
institute_counts = data['Institute Name'].value_counts()
valid_institutes = institute_counts[institute_counts >= 5].index
data = data[data['Institute Name'].isin(valid_institutes)]
after = len(data)
print(f"Removed {before - after} records from institutes with <5 samples")
print(f"Kept {len(valid_institutes)} institutes with sufficient data")

# Show summary statistics
print("\n" + "="*60)
print("CLEANED DATA SUMMARY")
print("="*60)
print(f"\nTotal records: {len(data)}")
print(f"\nUnique Institutes: {data['Institute Name'].nunique()}")
print(f"Unique Courses: {data['Course Name'].nunique()}")
print(f"Unique Locations: {data['Location'].nunique()}")

print("\n--- Locations ---")
print(data['Location'].value_counts().head(15))

print("\n--- Top 10 Institutes ---")
print(data['Institute Name'].value_counts().head(10))

print("\n--- Top 10 Courses ---")
print(data['Course Name'].value_counts().head(10))

# Save cleaned dataset
output_file = 'dataset_cleaned.csv'
data.to_csv(output_file, index=False)
print(f"\n✅ Cleaned dataset saved to: {output_file}")
print(f"✅ Ready for model training!")
