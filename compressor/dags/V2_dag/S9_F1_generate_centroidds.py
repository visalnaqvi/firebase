import psycopg2
from psycopg2.extras import DictCursor
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams, Distance
import logging
import os
import argparse
import time
from typing import Optional
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ==== CONFIG ====
PG_CONN = os.getenv("DATABASE_URL")  # PostgreSQL connection string
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")  # if using cloud Qdrant

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
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
                    WHERE task = 'centroid_generation'
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
                        WHERE task = 'centroid_generation'
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
    Updates process_status table where task = 'centroid_generation'
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
                        WHERE task = 'centroid_generation'
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
                        WHERE task = 'centroid_generation'
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
                        WHERE task = 'centroid_generation'
                        """,
                        (group_id,)
                    )
                    cur.execute(
                        """
                        UPDATE process_status
                        SET next_group_in_queue = %s, task_status = 'starting'
                        WHERE task = 'centroid_matching' and next_group_in_queue is null 
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
    # return psycopg2.connect(
    #     host="nozomi.proxy.rlwy.net",
    #     port="24794",
    #     dbname="railway",
    #     user="postgres",
    #     password="kdVrNTrtLzzAaOXzKHaJCzhmoHnSDKDG"
    # )
        

def get_warmed_groups():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=DictCursor)
    cur.execute("SELECT id FROM groups WHERE status = 'warmed' order by last_processed_at")
    result = [row["id"] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return result


def get_best_faces_for_group(group_id):
    try:
        """Get highest quality_score face per person_id for a group"""
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=DictCursor)
        cur.execute("""
            SELECT face_id as id , id as person_id, quality_score, thumbnail as face_thumb_bytes
            FROM persons
            WHERE group_id = %s
            and thumbnail is not null
        """, (group_id,))
        res = cur.fetchall()
        cur.close()
        conn.close()
        return res
    except Exception as e:
        raise



def ensure_person_centroid_collection(group_id, vector_size):
    """Ensure person_centroid_<group_id> exists"""
    try:
        collection_name = f"person_centroid_{group_id}"
        if qdrant.collection_exists(collection_name):
            qdrant.delete_collection(collection_name=collection_name)
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )
        logging.info(f"Created collection {collection_name}")
        return collection_name
    except Exception as e:
        raise

def process_group(group_id):
    try:
        best_faces = get_best_faces_for_group(group_id)
        if not best_faces:
            logging.info(f"No faces found for group {group_id}")
            return

        # Update persons table with best face thumbs

        # Get embedding size from first point
        first_face_id = best_faces[0]["id"]
        print("finding vector " + first_face_id)
        face_point = qdrant.retrieve(collection_name=str(group_id), ids=[first_face_id], with_vectors=True)
        if not face_point or not face_point[0].vector:
            logging.warning(f"No embedding found for first face in group {group_id}")
            return

        vector_data = face_point[0].vector
        if isinstance(vector_data, dict):
            vector_data = vector_data.get("face")  # Or whichever named vector you want
        vector_size = len(vector_data)
        target_collection = ensure_person_centroid_collection(group_id, vector_size)

        points_to_upsert = []
        for face in best_faces:
            face_id = face["id"]
            person_id = face["person_id"]

            retrieved = qdrant.retrieve(
                collection_name=str(group_id),
                ids=[face_id],
                with_vectors=True
            )
            if not retrieved or not retrieved[0].vector:
                logging.warning(f"No embedding found for face {face_id} in group {group_id}")
                continue

            vector_data = retrieved[0].vector
            if isinstance(vector_data, dict):
                vector_data = vector_data.get("face")  # pick only "face" vector

            points_to_upsert.append(
                PointStruct(
                    id=str(person_id),
                    vector=vector_data
                )
            )

        if points_to_upsert:
            qdrant.upsert(collection_name=target_collection, points=points_to_upsert)
            logging.info(f"Upserted {len(points_to_upsert)} person centroids into {target_collection}")
    except Exception as e:
        raise

def mark_group_processed(group_id) -> None:
    try:
        """Mark group_id as processed and clear image_byte"""
        if not group_id:
            return
            
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                query = """
                            UPDATE groups
                            SET status = 'cooling',
                    last_processed_at = NOW(),
                            last_processed_step = 'centroid'
                            WHERE id = %s AND status = 'warmed'
                        """
                cur.execute(query, (group_id,))
                conn.commit()
                print(f"Marked {group_id} group_id as processed")
    except Exception as e:
        raise
def main():
    try:
        group_id = get_or_assign_group_id()
        run_id = int(time.time())
        if not group_id:
            update_status(None , "no group to process" , True , "waiting")
            update_status_history(run_id , "centroid_generation" , "run" , None , None , None , None , "no_group")
            return False
        update_status(group_id , "running" , False , "healthy")
        update_status_history(run_id , "centroid_generation" , "run" , None , None , None , group_id , "started")

        logging.info(f"Processing group {group_id}")
        process_group(group_id)
        mark_group_processed(group_id)
        update_status(None , "done" , True , "done")
        update_status_history(run_id , "centroid_generation" , "run" , None , None , None , group_id , "done")
        update_last_provrssed_group_column(group_id)
        return True
    except Exception as e:
        update_status(group_id , f"error while centroid generation {e}" , True , "failed")
        update_status_history(run_id , "centroid_generation" , "run" , None , None , None , group_id , f"error {e}")

        logging.info(f"failed group {group_id}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
