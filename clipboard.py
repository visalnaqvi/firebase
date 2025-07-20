import pyperclip
import time
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Firebase
cred = credentials.Certificate("key.json")  # Replace with your key file path
firebase_admin.initialize_app(cred)
db = firestore.client()

# Track last clipboard content
last_text = ""

while True:
    try:
        text = pyperclip.paste()
        if text != last_text:
            last_text = text
            print("Clipboard changed:", text)

            # Insert new document into Firestore
            doc_ref = db.collection('clipboard').add({
                'content': text,
                'timestamp': firestore.SERVER_TIMESTAMP
            })
            print(f"Inserted into Firestore: {doc_ref}")

    except Exception as e:
        print("Error:", e)

    time.sleep(1)  # Check every second
