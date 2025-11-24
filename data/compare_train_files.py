import json
import os

# Read all training files
with open('train_algebra.json', 'r', encoding='utf-8') as f:
    algebra_data = json.load(f)

with open('train_history.json', 'r', encoding='utf-8') as f:
    history_data = json.load(f)

with open('train_geography.json', 'r', encoding='utf-8') as f:
    geography_data = json.load(f)

with open('train_chinese.json', 'r', encoding='utf-8') as f:
    chinese_data = json.load(f)

with open('train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# Create sets of questionIDs from each file
algebra_ids = {item['questionID'] for item in algebra_data}
history_ids = {item['questionID'] for item in history_data}
geography_ids = {item['questionID'] for item in geography_data}
chinese_ids = {item['questionID'] for item in chinese_data}
train_ids = {item['questionID'] for item in train_data}

# Find all IDs that should be in train.json
all_subject_ids = algebra_ids | history_ids | geography_ids | chinese_ids

# Find missing IDs
missing_ids = all_subject_ids - train_ids

print(f"Algebra entries: {len(algebra_data)}")
print(f"History entries: {len(history_data)}")
print(f"Geography entries: {len(geography_data)}")
print(f"Chinese entries: {len(chinese_data)}")
print(f"Total subject-specific entries: {len(all_subject_ids)}")
print(f"Train.json entries: {len(train_data)}")
print(f"\nMissing questionIDs in train.json: {len(missing_ids)}")
if missing_ids:
    print(f"Missing IDs: {sorted(missing_ids)}")

# Create a mapping of questionID to full entry for easy lookup
all_entries = {}
for item in algebra_data:
    all_entries[item['questionID']] = item
for item in history_data:
    all_entries[item['questionID']] = item
for item in geography_data:
    all_entries[item['questionID']] = item
for item in chinese_data:
    all_entries[item['questionID']] = item

# Get missing entries
missing_entries = [all_entries[qid] for qid in missing_ids if qid in all_entries]

print(f"\nMissing entries to add: {len(missing_entries)}")

# Save missing entries to a file for review
if missing_entries:
    with open('missing_entries.json', 'w', encoding='utf-8') as f:
        json.dump(missing_entries, f, ensure_ascii=False, indent=2)
    print("Missing entries saved to missing_entries.json")

