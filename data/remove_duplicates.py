import json

# Read train.json
with open('train.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Remove duplicates, keeping first occurrence
seen = set()
unique_data = []
for item in data:
    qid = item['questionID']
    if qid not in seen:
        seen.add(qid)
        unique_data.append(item)

# Sort by questionID
unique_data.sort(key=lambda x: x['questionID'])

# Write back
with open('train.json', 'w', encoding='utf-8') as f:
    json.dump(unique_data, f, ensure_ascii=False, indent=4)

print(f'Removed duplicates. Final count: {len(unique_data)}')

