import json

# Read current train.json
with open('train.json', 'r', encoding='utf-8') as f:
    train_data = json.load(f)

# Read missing entries
with open('missing_entries.json', 'r', encoding='utf-8') as f:
    missing_entries = json.load(f)

# Create a set of existing questionIDs to avoid duplicates
existing_ids = {item['questionID'] for item in train_data}

# Add missing entries
added_count = 0
for entry in missing_entries:
    if entry['questionID'] not in existing_ids:
        train_data.append(entry)
        existing_ids.add(entry['questionID'])
        added_count += 1

# Sort by questionID for consistency
train_data.sort(key=lambda x: x['questionID'])

# Write updated train.json
with open('train.json', 'w', encoding='utf-8') as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

print(f"Added {added_count} missing entries to train.json")
print(f"Total entries in train.json: {len(train_data)}")

