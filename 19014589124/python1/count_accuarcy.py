#!/usr/bin/env python3
import pandas as pd
import sys

# Check if file path is provided
if len(sys.argv) < 2:
    print("Usage: python count_accuracy.py <csv_file_path>")
    sys.exit(1)

file_path = sys.argv[1]

# Read CSV using pandas
try:
    df = pd.read_csv(file_path)
except Exception as e:
    print(f"Error reading file: {e}")
    sys.exit(1)

# Check if 'accuracy' column exists
if 'accuracy' not in df.columns:
    print("Column 'accuracy' not found in the CSV.")
    sys.exit(1)

# Count occurrences of accuracy values
accuracy_counts = df['accuracy'].value_counts()

print("\nAccuracy Counts:")
print(accuracy_counts)
