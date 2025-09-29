#!/bin/bash
set -e  # Exit immediately if any command fails

# Activate virtual environment (Git Bash style)
source /c/Users/ayatm/OneDrive/Desktop/visal/firebase/compressor/venv_fixed/Scripts/activate
# C:\Users\ayatm\OneDrive\Desktop\visal\firebase\compressor\venv_fixed
# Infinite loop
while true; do
    echo "S4_update_group_heating_to_warm..."
    python S4_update_group_heating_to_warm.py

    echo "Assigning next task to extraction..."
    python assign_task_to_extraction.py

    echo "Processing group running extraction"
    python S4_V2_people_extract_and_emb_generation.py 

    echo "Processing group running quality score"
    python S5_V2_assign_face_quality_score.py

    echo "Processing group running grouping"
    python S6_V2_group_emb_using_db.py

    echo "Processing group running insertion"
    python S7_V2_insert_person_table.py

    echo "Processing group running thumbnail insertion"
    python S8_update_thumbnail.py

    echo "Processing group running centroid generation"
    python S9_F1_generate_centroidds.py

    echo "Processing group running centroid matching"
    python S10_F1_centroid_matching.py

    echo "✅ Group processed successfully."
done

# Deactivate venv after finishing
deactivate
