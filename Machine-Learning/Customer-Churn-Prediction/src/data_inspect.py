import os
import pandas as pd

# Build an absolute path (works from any folder)
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(base_dir, "data", "telecom.csv")

# Load dataset
df = pd.read_csv(file_path)
print("✅ Dataset loaded successfully from:", file_path)
print("Shape:", df.shape)
print(df.head())



