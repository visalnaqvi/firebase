import os
import json
import psycopg2
from psycopg2.extras import execute_values
from collections import defaultdict
from typing import Optional
import time
class ProcessingError(Exception):
    def __init__(self, message, group_id=None, reason=None, retryable=True):
        super().__init__(message)
        self.group_id = group_id
        self.reason = reason
        self.retryable = retryable

    def __str__(self):
        return f"ProcessingError: {self.args[0]} (group_id={self.group_id}, reason={self.reason}, retryable={self.retryable})"
def get_or_assign_group_id():
    """
    Fetch the active group_id for extraction task.
    - If processing_group has a value → return it
    - Else if next_group_in_queue has a value → move it to processing_group,
    set next_group_in_queue = NULL, return it
    - Else return None
    """
    conn = None
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Fetch both columns
                cur.execute(
                    """
                    SELECT processing_group, next_group_in_queue
                    FROM process_status
                    WHERE task = 'insertion'
                    LIMIT 1
                    """
                )
                row = cur.fetchone()

                if not row:
                    return None

                processing_group, next_group_in_queue = row

                if processing_group:
                    return processing_group

                if next_group_in_queue:
                    # Promote next_group_in_queue → processing_group
                    cur.execute(
                        """
                        UPDATE process_status
                        SET processing_group = %s,
                            next_group_in_queue = NULL
                        WHERE task = 'insertion'
                        """,
                        (next_group_in_queue,)
                    )
                    conn.commit()
                    return next_group_in_queue

                return None
    except Exception as e:
        print("❌ Error in get_or_assign_group_id:", e)
        return None


def update_status_history(
    run_id: int,
    task: str,
    sub_task: str,
    totalImagesInitialized: int,
    totalImagesFailed: int,
    totalImagesProcessed: int,
    groupId: Optional[str],
    fail_reason: Optional[str]
) -> bool:
    """
    Insert a record into process_history.
    Returns True if insert succeeded, False otherwise.
    """

    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO process_history
                        (worker_id, run_id, task, sub_task,
                        initialized_count, success_count, failed_count,
                        group_id, ended_at, fail_reason)
                    VALUES
                        (%s, %s, %s, %s,
                        %s, %s, %s,
                        %s, NOW(), %s)
                    """,
                    (
                        1,                       # worker_id
                        run_id,
                        task,
                        sub_task,
                        totalImagesInitialized,
                        totalImagesProcessed,
                        totalImagesFailed,
                        groupId,
                        fail_reason,
                    )
                )
                conn.commit()
                return True
    except Exception as e:
        print(f"❌ Error inserting into process_history: {e}")
        return False
    
    
def update_status(group_id, fail_reason, is_ideal , status):
    """
    Updates process_status table where task = 'insertion'
    Returns a dict with success flag and optional error.
    """
    conn = None

    
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                if (status=='failed'):
                    cur.execute(
                        """
                        UPDATE process_status
                        SET task_status = %s,
                            fail_reason = %s,
                            ended_at = NOW(),
                            is_ideal = %s
                        WHERE task = 'insertion'
                        """,
                        (status , fail_reason, is_ideal)
                    )
                else:
                    cur.execute(
                        """
                        UPDATE process_status
                        SET task_status = %s,
                            processing_group = %s,
                            fail_reason = %s,
                            ended_at = NOW(),
                            is_ideal = %s
                        WHERE task = 'insertion'
                        """,
                        (status , group_id, fail_reason, is_ideal)
                    )
            conn.commit()
            return {"success": True}
    except Exception as e:
        print("❌ Error updating process status:", e)
        if conn:
            conn.rollback()
        return {"success": False, "errorReason": "updating status", "error": str(e)}
    finally:
        if conn:
            conn.close()
            
def update_last_provrssed_group_column(group_id):
        """
        Updates process_status table where task = 'extraction'
        Returns a dict with success flag and optional error.
        """
        conn = None
        
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE process_status
                        SET last_group_processed = %s
                        WHERE task = 'insertion'
                        """,
                        (group_id,)
                    )
                    cur.execute(
                        """
                        UPDATE process_status
                        SET next_group_in_queue = %s
                        WHERE task = 'thumbnail' and next_group_in_queue is null 
                        """,
                        (group_id,)
                    )
                    if cur.rowcount == 0:
                            raise Exception("No rows updated for quality_assignment (next_group_in_queue was not NULL)")
                conn.commit()
                return {"success": True}
        except Exception as e:
            print("❌ Error updating process status:", e)
            if conn:
                conn.rollback()
            return {"success": False, "errorReason": "updating status", "error": str(e)}
        finally:
            if conn:
                conn.close()
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
    run_id = int(time.time())
    if not group_id:
        update_status(None , "No group to process" , True , "waiting")
        update_status_history(run_id , "insertion" , "group" , None , None  , None , group_id , "no_group Found")
        return True
   
    conn = get_db_connection()
    try:
        update_status(group_id , "running" , False , "healthy")
        update_status_history(run_id , "insertion" , "group" , None , None  , None , group_id , "started")
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
        update_status(None , "" , True , "done")
        update_status_history(run_id , "insertion" , "group" , None , None  , None , group_id , "done")
        update_last_provrssed_group_column(group_id)
    except Exception as e:
        print(f"[WARNING] Error in sync_single_group: {e}")
        conn.rollback()
        update_status(group_id , f"Error while trying insertion : {e}" , True , "failed")
        update_status_history(run_id , "insertion" , "group" , None , None  , None , group_id , f"error while trying insertion : {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    # Sync all warmed groups
    # sync_persons()
   

    group_id = get_or_assign_group_id()
    # Or sync a single group for testing:
    success = sync_single_group(group_id)
    exit(0 if success else 1)