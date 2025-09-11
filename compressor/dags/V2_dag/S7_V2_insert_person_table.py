import os
import json
import psycopg2
from psycopg2.extras import execute_values
from collections import defaultdict
import argparse

def get_db_connection():
    return psycopg2.connect(
        host="ballast.proxy.rlwy.net",
        port="56193",
        dbname="railway",
        user="postgres",
        password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
    )
    
def check_group_exists(group_id: int) -> bool:
    """Check if group exists and has warm status"""
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM groups WHERE id = %s AND status = 'warmed' and last_processed_step='grouping'", (group_id,))
            result = cur.fetchone()
            return result is not None
def mark_group_process_status(group_id , status) -> None:
            """Mark group_id as being processed"""
            if not group_id:
                return
                
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    query = """
                        UPDATE process_status
                        SET group_id = %s,
                        running = 'insertion',
                        status = %s,
                        started_at = NOW()
                        WHERE id = 1
                    """
                    cur.execute(query, (group_id,status))
                    conn.commit()
def load_faces_from_json(group_id):
    """Load faces data from JSON file"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(script_dir, "warm-images", str(group_id), "faces", "faces.json")
    
    if not os.path.exists(json_path):
        print(f"[WARNING] JSON file not found: {json_path}")
        return []
    
    try:
        with open(json_path, 'r') as f:
            faces_data = json.load(f)
        
        print(f"[SUCCESS] Loaded {len(faces_data)} faces from JSON file")
        return faces_data
        
    except Exception as e:
        print(f"[WARNING] Error loading JSON file: {e}")
        return []

def bulk_insert_faces_to_db(conn, group_id, faces_data):
    """Bulk insert faces data into faces table"""
    if not faces_data:
        return
    
    cur = conn.cursor()
    
    try:
        # Prepare data for bulk insert
        # Assuming faces table structure includes: id, image_id, person_id, quality_score, insight_face_confidence, group_id
        insert_data = []
        for face in faces_data:
            insert_data.append((
                face.get('id'),
                face.get('image_id'),
                face.get('person_id'),
                face.get('quality_score'),
                face.get('insight_face_confidence'),
                group_id,
                face.get('status', 'processed')  # Default status if not present
            ))
        
        # Bulk insert using execute_values for better performance
        execute_values(
            cur,
            """
            INSERT INTO faces (id, image_id, person_id, quality_score, insight_face_confidence, group_id, status)
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                person_id = EXCLUDED.person_id,
                quality_score = EXCLUDED.quality_score,
                insight_face_confidence = EXCLUDED.insight_face_confidence,
                status = EXCLUDED.status
            """,
            insert_data,
            template=None,
            page_size=1000
        )
        
        conn.commit()
        print(f"[SUCCESS] Bulk inserted/updated {len(insert_data)} faces into database")
        
    except Exception as e:
        conn.rollback()
        print(f"[WARNING] Error bulk inserting faces: {e}")
        raise e
    finally:
        cur.close()

def create_persons_from_json(conn, group_id, faces_data):
    """Create persons directly from JSON data for better performance"""
    if not faces_data:
        return
    
    cur = conn.cursor()
    
    try:
        # Group faces by person_id and find best quality face for each person
        person_data = defaultdict(list)
        
        for face in faces_data:
            person_id = face.get('person_id')
            if person_id:  # Only process faces with assigned person_id
                person_data[person_id].append(face)
        
        if not person_data:
            print(f"[WARNING] No faces with person_id found for group {group_id}")
            return
        
        # Prepare persons insert data
        persons_insert_data = []
        
        for person_id, faces in person_data.items():
            # Find face with highest quality score
            best_face = max(faces, key=lambda x: x.get('quality_score', 0))
            total_images = len(faces)
            
            persons_insert_data.append((
                person_id,                           # id
                None,                               # thumbnail (will be populated later if needed)
                None,                               # name
                None,                               # user_id
                None,                               # image_ids (now always NULL)
                group_id,                           # group_id
                best_face.get('quality_score', 0),  # quality_score
                best_face.get('id'),                # face_id
                total_images                        # total_images
            ))
        
        # Bulk insert persons
        execute_values(
            cur,
            """
            INSERT INTO persons (id, thumbnail, name, user_id, image_ids, group_id, quality_score, face_id, total_images)
            VALUES %s
            ON CONFLICT (id) DO UPDATE SET
                total_images = EXCLUDED.total_images,
                group_id = EXCLUDED.group_id,
                quality_score = EXCLUDED.quality_score,
                face_id = EXCLUDED.face_id
            """,
            persons_insert_data,
            template=None,
            page_size=1000
        )
        
        conn.commit()
        print(f"[SUCCESS] Created/updated {len(persons_insert_data)} persons from JSON data")
        
    except Exception as e:
        conn.rollback()
        print(f"[WARNING] Error creating persons from JSON: {e}")
        raise e
    finally:
        cur.close()

def create_persons_from_db(conn, group_id):
    """Alternative: Create persons using database queries after faces are inserted"""
    cur = conn.cursor()
    
    try:
        # Use the existing SQL logic but with faces table
        cur.execute(
            """
            INSERT INTO persons (id, thumbnail, name, user_id, image_ids, group_id, quality_score, face_id, total_images)
            WITH best_faces AS (
                SELECT DISTINCT ON (person_id)
                    person_id,
                    face_thumb_bytes,
                    quality_score,
                    id
                FROM faces
                WHERE group_id = %s
                  AND person_id IS NOT NULL
                  AND status != 'error'
                ORDER BY person_id, quality_score DESC NULLS LAST
            ),
            image_counts AS (
                SELECT person_id, COUNT(*) AS total_images
                FROM faces
                WHERE group_id = %s
                  AND person_id IS NOT NULL
                  AND status != 'error'
                GROUP BY person_id
            )
            SELECT 
                b.person_id,
                b.face_thumb_bytes,
                NULL,   -- name placeholder
                NULL,   -- user_id placeholder
                NULL,   -- image_ids now always NULL
                %s as group_id,
                b.quality_score,
                b.id as face_id,
                c.total_images
            FROM best_faces b
            JOIN image_counts c ON b.person_id = c.person_id
            ON CONFLICT (id) DO UPDATE
            SET total_images = EXCLUDED.total_images,
                group_id = EXCLUDED.group_id,
                quality_score = EXCLUDED.quality_score,
                face_id = EXCLUDED.face_id
            """,
            (group_id, group_id, group_id)
        )
        
        conn.commit()
        affected_rows = cur.rowcount
        print(f"[SUCCESS] Created/updated {affected_rows} persons using database queries")
        
    except Exception as e:
        conn.rollback()
        print(f"[WARNING] Error creating persons from database: {e}")
        raise e
    finally:
        cur.close()

def sync_persons():
    conn = get_db_connection()

    try:
        # 1 Get all warmed groups
        cur = conn.cursor()
        cur.execute("SELECT id FROM groups WHERE status = 'warmed'")
        group_ids = [row[0] for row in cur.fetchall()]
        cur.close()

        print(f"Found {len(group_ids)} warmed groups: {group_ids}")

        for group_id in group_ids:
            print(f"\n[PROCESSING] Processing group {group_id}")

            # 2 Load faces data from JSON file
            faces_data = load_faces_from_json(group_id)
            
            if not faces_data:
                print(f"[WARNING] No faces data found for group {group_id}, skipping...")
                continue

            # 3 Bulk insert faces into database
            print(f"   Inserting faces into database...")
            bulk_insert_faces_to_db(conn, group_id, faces_data)

            # 4 Create persons - Choose one of two approaches:
            
            # Approach A: Create persons directly from JSON data (faster, more efficient)
            print(f"    Creating persons from JSON data...")
            create_persons_from_json(conn, group_id, faces_data)
            
            # Approach B: Create persons using database queries (uncomment if preferred)
            # print(f"    Creating persons from database...")
            # create_persons_from_db(conn, group_id)

        print("\n[SUCCESS] Sync complete for all groups")

    except Exception as e:
        print(f"[WARNING] Error in sync_persons: {e}")
        conn.rollback()

    finally:
        conn.close()

def sync_single_group(group_id):
    """Sync persons for a single group (useful for testing)"""
    if not check_group_exists(group_id):
        print(f"Group {group_id} not found")
        raise
    conn = get_db_connection()
    mark_group_process_status(group_id, 'healthy')
    try:
        print(f"[PROCESSING] Processing single group {group_id}")

        # Load faces data from JSON file
        faces_data = load_faces_from_json(group_id)
        
        if not faces_data:
            print(f"[WARNING] No faces data found for group {group_id}")
            return

        # Bulk insert faces into database
        print(f"Inserting faces into database...")
        bulk_insert_faces_to_db(conn, group_id, faces_data)

        # Create persons from JSON data
        print(f"Creating persons from JSON data...")
        create_persons_from_json(conn, group_id, faces_data)

        print("[SUCCESS] Single group sync complete")
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE groups 
                SET last_processed_step = 'insertion' 
                WHERE id = %s
            """, (group_id,))
            conn.commit()
        mark_group_process_status(group_id, 'done')
    except Exception as e:
        print(f"[WARNING] Error in sync_single_group: {e}")
        mark_group_process_status(group_id, 'failed')
        conn.rollback()

    finally:
        conn.close()

if __name__ == "__main__":
    # Sync all warmed groups
    # sync_persons()
    parser = argparse.ArgumentParser(description="Process face quality for a specific group.")
    parser.add_argument("group_id", type=int, help="Group ID to process")
    args = parser.parse_args()

    group_id = args.group_id

    # Or sync a single group for testing:
    sync_single_group(group_id)