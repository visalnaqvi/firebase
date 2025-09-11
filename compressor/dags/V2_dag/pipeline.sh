#!/bin/bash
set -e
# Activate virtual environment (Git Bash style)
source /c/Users/visal.naqvi/Desktop/DB-Data-backup/mywork-space/firebase/compressor/venv/Scripts/activate
echo "Running S2"
python S2_A_update_groups_last_image_timestamp.py
echo "Running S3"
python S3_update_group_heating_to_warm.py
echo "Running S3 V2"
group_ids=$(python S3_V2_get_warm_groups.py)

if [ -z "$group_ids" ]; then
  echo "No warm groups found."
  deactivate
  exit 0
fi

# Loop through IDs
for gid in $group_ids; do
    echo "Processing group running extraction: $gid"
    python S4_V2_people_extract_and_emb_generation.py "$gid"

    echo "Processing group running quality score: $gid"
    python S5_V2_assign_face_quality_score.py "$gid"

    echo "Processing group running grouping: $gid"
    python S6_V2_group_emb_using_db.py "$gid"

    echo "Processing group running insertion: $gid"
    python S7_V2_insert_person_table.py "$gid"

    echo "Processing group running thumbnail insertion: $gid"
    python S8_update_thumbnail.py "$gid"

    echo "Processing group running centroid generation: $gid"
    python S9_F1_generate_centroidds.py "$gid"

    echo "Processing group running centroid matching: $gid"
    python S10_F1_centroid_matching.py "$gid"

    if [ $? -ne 0 ]; then
        echo "❌ Error processing group $gid, stopping loop."
        deactivate
        exit 1
    fi
done

echo "✅ All groups processed successfully."

# Deactivate venv after finishing
deactivate
