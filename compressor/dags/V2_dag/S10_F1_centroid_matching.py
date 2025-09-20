import psycopg2
from psycopg2.extras import execute_values
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, SearchRequest
import numpy as np
import logging
import argparse
import time
from typing import Optional
# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
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
                    WHERE task = 'centroid_matching'
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
                        WHERE task = 'centroid_matching'
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
    Updates process_status table where task = 'centroid_matching'
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
                        WHERE task = 'centroid_matching'
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
                        WHERE task = 'centroid_matching'
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
                        WHERE task = 'centroid_matching'
                        """,
                        (group_id,)
                    )
                    cur.execute(
                        """
                        UPDATE process_status
                        SET last_group_processed = %s, task_status = 'starting'
                        WHERE task = 'master' and next_group_in_queue is null 
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
# DB connection
def get_db_connection():
    # return psycopg2.connect(
    #      host="ballast.proxy.rlwy.net",
    #     port="56193",
    #     dbname="railway",
    #     user="postgres",
    #     password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
    # )
    return psycopg2.connect(
        host="nozomi.proxy.rlwy.net",
        port="24794",
        dbname="railway",
        user="postgres",
        password="kdVrNTrtLzzAaOXzKHaJCzhmoHnSDKDG"
    )

                    
# Get distinct person_id, group_id pairs
def get_unique_persons(group_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT person_id
            FROM faces
            WHERE person_id IS NOT NULL and group_id = %s
        """ , (group_id,))
        results = cur.fetchall()
        conn.close()
        logger.info(f"Retrieved {len(results)} unique (person_id, group_id) pairs")
        return [item[0] for item in results]
    except Exception as e:
        raise
# Fetch vector for given person from Qdrant
def get_person_vector(qdrant_client, collection_name, person_id):
    try:
        points = qdrant_client.retrieve(
            collection_name=collection_name,
            ids=[person_id],
            with_vectors=True
        )
        if points and points[0].vector is not None:
            vec = points[0].vector
            if isinstance(vec, dict):  # Named vector handling
                vec = next(iter(vec.values()))
            return np.array(vec, dtype=np.float32)
    except Exception as e:
        logger.error(f"Error retrieving vector for person {person_id} from {collection_name}: {e}")
        raise

# Find similar candidates above threshold
def find_similar(qdrant_client, collection_name, vector, person_id, threshold, top_k=10):
    try:
        results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=vector.tolist(),
            limit=top_k,
            score_threshold=threshold
        )
        similar_ids = []
        for r in results:
            if r.id != person_id and r.score >= threshold:
                similar_ids.append(r.id)
        return similar_ids
    except Exception as e:
        logger.error(f"Error searching similar persons for {person_id} in {collection_name}: {e}")
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
                    set last_processed_at = NOW(),
                            last_processed_step = 'similarity'
                            WHERE id = %s AND status = 'cooling'
                        """
                cur.execute(query, (group_id,))
                conn.commit()
                print(f"Marked {group_id} group_id as processed")
    except Exception as e:
        raise
# Insert into similar_faces table
def insert_similar_faces(pairs):
    try:
        if not pairs:
            return
        conn = get_db_connection()
        cur = conn.cursor()
        execute_values(cur, """
            INSERT INTO similar_faces (person_id, similar_person_id, group_id)
            VALUES %s
        """, pairs)
        conn.commit()
        conn.close()
        logger.info(f"Inserted {len(pairs)} similar face records")
    except Exception as e:
        raise

def main():
    
    
    try:
        
        group_id = get_or_assign_group_id()
        run_id = int(time.time())
        if not group_id:
            update_status(None , "no group to process" , True , "waiting")
            update_status_history(run_id , "centroid_matching" , "run" , None , None , None , None , "no_group")
            return False
        update_status(group_id , "running" , False , "healthy")
        update_status_history(run_id , "centroid_matching" , "run" , None , None , None , group_id , "started")
        qdrant_client = QdrantClient(host="localhost", port=6333)

        persons = get_unique_persons(group_id)
        all_pairs = []
        for person_id in persons:
            collection_name = f"person_centroid_{group_id}"
            vector = get_person_vector(qdrant_client, collection_name, person_id)
            if vector is None:
                logger.warning(f"No vector found for person_id={person_id} in group_id={group_id}")
                continue

            similar_ids = find_similar(qdrant_client, collection_name, vector, person_id, threshold=0.3)
            for sim_id in similar_ids:
                all_pairs.append((person_id, sim_id,group_id ))

        insert_similar_faces(all_pairs)
        mark_group_processed(group_id)
        update_status(None , "done" , True , "done")
        update_status_history(run_id , "centroid_matching" , "run" , None , None , None , group_id , "done")
        update_last_provrssed_group_column(group_id)
        logger.info(f"Similarity check completed for group {group_id}")
        logger.info("Similarity check completed")
        return True
    except Exception as e:
        logger.info(f"Similarity failed for group {group_id}")
        update_status(group_id , f"error while centroid generation {e}" , True , "failed")
        update_status_history(run_id , "centroid_matching" , "run" , None , None , None , group_id , f"error {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
