import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd

# Step 1: Initialize Firebase App
cred = credentials.Certificate("key.json")  # Replace with your key file path
firebase_admin.initialize_app(cred)

# Step 2: Access Firestore
db = firestore.client()

# Step 3: Get Documents from Firestore
collection_name = 'registrations-ssc'
docs = db.collection(collection_name).stream()

# Step 4: Convert Documents to List of Dicts
data = []
for doc in docs:
    doc_dict = doc.to_dict()
    data.append(doc_dict)

# Step 5: Convert to DataFrame
df = pd.DataFrame(data)

# Step 6: Extract 'date' from 'timestamp'
df['date'] = pd.to_datetime(df['timestamp'], errors='coerce').dt.strftime('%Y/%m/%d')

# Step 7: Count how many times each gclid appeared (excluding null gclids)
df['gclid_count'] = df.groupby('gclid')['gclid'].transform('count')

# Step 8: Drop duplicate gclids, keeping the first one
# For NaN gclids (missing), they are kept as separate since they can't be grouped
df_deduped = df.sort_values('timestamp').drop_duplicates(subset=['gclid'], keep='first')

# Step 9: Save to Excel
df_deduped.to_excel("ssc_firebase_leads.xlsx", index=False)

print("Excel file saved as firebase_leads.xlsx with 1 row per gclid.")
