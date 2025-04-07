#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/**
 * @file preprocess_rdap.py
 * @brief Preprocess and flatten rdap JSON data from MongoDB.
 *
 * This script reads a JSON file (exported from MongoDB) where each line contains
 * a document with fields such as "domain_name" and a nested "rdap" object. It then:
 *   - Loads the JSON data line by line.
 *   - Flattens the nested "rdap" object into individual columns using pandas.json_normalize.
 *   - Drops the "_id" field from the data.
 *   - Optionally renames the flattened columns by removing the "rdap." prefix.
 *
 * The resulting pandas DataFrame consists of a "domain_name" column alongside all the
 * fields from the "rdap" object, making it ready for further data analysis.
 *
 * Usage:
 *   Run the script to generate the DataFrame and inspect or process the data as needed.
 *
 * @author Filip Bucko
 * @date 30.3.2024
 */
"""

import json
import pandas as pd
from pathlib import Path

# Construct the path to the JSON file relative to this script's location.
# __file__ is the path to this script (transformers/data/our_script.py)
script_path = Path(__file__).resolve()
# Go up two levels to the project root (from transformers/data -> transformers -> project_root)
project_root = script_path.parent.parent.parent
# Now build the path to the JSON file in datasets/malware/
json_file_path = project_root / "datasets" / "phishing" / "phishing_strict_rdap_2024.json"

# Read the JSON file line by line
with open(json_file_path, 'r') as f:
    data = [json.loads(line) for line in f]

# Flatten the JSON structure; each nested key from 'rdap' becomes a column like 'rdap.handle', etc.
df = pd.json_normalize(data, sep='.')

# Drop any _id columns (they typically appear as _id.$oid)
df = df.drop(columns=[col for col in df.columns if col.startswith('_id')])

# Optionally remove the "rdap." prefix from column names for clarity
df.columns = [col.replace('rdap.', '') for col in df.columns]

# Now, df contains the "domain_name" column and all flattened rdap fields.
print(df.head())
