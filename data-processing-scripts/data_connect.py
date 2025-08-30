"""
Data Connect Script - Merges review and meta JSON files for a given state

FOLDER SETUP:
Create a folder called "data", and copy and update your base_path on line 40.
Place your JSON files directly in the base_path directory with the following naming convention:
- review-[StateName].json (e.g., review-Alaska.json)
- meta-[StateName].json (e.g., meta-Alaska.json)

USAGE:
1. Command line: python3 data_connect.py [state_name]
   Example: python3 data_connect.py alaska
   
2. The script will:
   - Load both JSON files (JSONL format)
   - Filter reviews to only include those with matching business records
   - Sample up to 100 records (configurable on line 55)
   - Merge the data on gmap_id
   - Output a merged-[StateName].json file in the same directory

OUTPUT:
- Creates merged-[StateName].json in JSONL format
- Preserves original data types and structure
"""

import pandas as pd
import sys
import os

if len(sys.argv) > 1:
    state = sys.argv[1].lower()
else:
    print("Usage: python3 data_connect.py [state_name]")
    print("Example: python3 data_connect.py alaska")
    print("\nEnsure your JSON files are in the base directory with naming:")
    print("- review-[StateName].json")
    print("- meta-[StateName].json")
    sys.exit(1)

base_path = "/Users/gsharsh/Downloads/data"
state_title = state.title()
review_file = f"{base_path}/review-{state_title}.json"
meta_file = f"{base_path}/meta-{state_title}.json"

print(f"Processing {state_title} data...")

df1 = pd.read_json(review_file, lines=True, dtype={"user_id": "string"})
df2 = pd.read_json(meta_file, lines=True)

print(f"Loaded {len(df1)} reviews and {len(df2)} business records")

df1_filtered = df1[df1['gmap_id'].isin(df2['gmap_id'])]
print(f"Filtered to {len(df1_filtered)} reviews with matching business records")

sample_size = min(100, len(df1_filtered))
sampled_df1 = df1_filtered.sample(n=sample_size, random_state=42)
print(f"Sampled {len(sampled_df1)} reviews")

merged_df = pd.merge(sampled_df1, df2, on='gmap_id', how='left')
print(f"Merged data has {len(merged_df)} records")

output_file = f"merged-{state_title}.json"
merged_df.to_json(output_file, orient="records", lines=True)

print(f"Merged JSONL file created: {output_file}")
print(f"Total records: {len(merged_df)}")
print("Format: JSONL (one JSON object per line) with preserved data types")
