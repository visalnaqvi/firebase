import os
import psycopg2
import argparse
import logging
import time 
from typing import Optional
# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
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
                    WHERE task = 'thumbnail'
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
                        WHERE task = 'thumbnail'
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
                        WHERE task = 'thumbnail'
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
                        WHERE task = 'thumbnail'
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
                        WHERE task = 'thumbnail'
                        """,
                        (group_id,)
                    )
                    cur.execute(
                        """
                        UPDATE process_status
                        SET next_group_in_queue = %s
                        WHERE task = 'centroid_generation' and next_group_in_queue is null 
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
# -------------------- DB Connection --------------------
def get_db_connection():
    return psycopg2.connect(
        host="ballast.proxy.rlwy.net",
        port="56193",
        dbname="railway",
        user="postgres",
        password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
    )

        
# -------------------- Main Script --------------------
def process_group(group_id: int):
    
    try:
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "warm-images", str(group_id), "faces")

        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Fetch all persons with group_id
                cur.execute("SELECT id, face_id FROM persons WHERE group_id = %s", (group_id,))
                rows = cur.fetchall()

                if not rows:
                    logger.warning(f"No persons found for group {group_id}")
                    raise ProcessingError("No Persons found for this group")

                logger.info(f"Found {len(rows)} records for group {group_id}")

                for person_id, face_id in rows:
                    image_path = os.path.join(base_path, f"{face_id}.jpg")

                    if not os.path.exists(image_path):
                        logger.error(f"Image not found: {image_path} (person_id={person_id})")
                        raise ProcessingError(f"Image not found: {image_path} (person_id={person_id})")

                    try:
                        with open(image_path, "rb") as f:
                            image_bytes = f.read()

                        cur.execute(
                            """
                            UPDATE persons
                            SET thumbnail = %s
                            WHERE id = %s
                            """,
                            (psycopg2.Binary(image_bytes), person_id)
                        )
                        logger.info(f"Updated thumbnail for person_id={person_id}, face_id={face_id}")

                    except Exception as e:
                        logger.error(f"Failed to update person_id={person_id}, face_id={face_id}: {e}")
                        raise

                conn.commit()
                logger.info(f" Completed processing group {group_id}")
    except Exception as e:
        raise
def main():
    try:
        run_id = int(time.time())
        group_id = get_or_assign_group_id()
        if not group_id:
            update_status(group_id , f"no group to prcess" , True , "waiting")
            update_status_history(run_id , "thumbnail" , "run" , None , None , None , group_id , f"no_group")

        update_status(group_id , f"running" , False , "healthy")
        update_status_history(run_id , "thumbnail" , "run" , None , None , None , group_id , f"started")

        process_group(group_id)
        
        with get_db_connection() as conn, conn.cursor() as cur:
                cur.execute("""
                    UPDATE groups 
                    SET last_processed_step = 'thumbnail_insertion' 
                    WHERE id = %s
                """, (group_id,))
                conn.commit()
        update_status(None , f"done" , True , "done")
        update_status_history(run_id , "thumbnail" , "run" , None , None , None , group_id , f"done")
        update_last_provrssed_group_column(group_id)
        return True
    except Exception as e:
        update_status(group_id , f"error occured in thumbnail : {e}" , True , "failed")
        update_status_history(run_id , "thumbnail" , "run" , None , None , None , group_id , f"error in thumbnail {e}")
        return False
if __name__ == "__main__":


    success = main()
    exit(0 if success else 1)
