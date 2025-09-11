import os
import psycopg2
import argparse
import logging

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# -------------------- DB Connection --------------------
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
            cur.execute("SELECT id FROM groups WHERE id = %s AND status = 'warmed' and last_processed_step='insertion'", (group_id,))
            result = cur.fetchone()
            return result is not None
# -------------------- Main Script --------------------
def process_group(group_id: int):
    base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "warm-images", str(group_id), "faces")
    if not check_group_exists(group_id):
        print(f"Group {group_id} not found")
        raise
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # Fetch all persons with group_id
            cur.execute("SELECT id, face_id FROM persons WHERE group_id = %s", (group_id,))
            rows = cur.fetchall()

            if not rows:
                logger.warning(f"No persons found for group {group_id}")
                return

            logger.info(f"Found {len(rows)} records for group {group_id}")

            for person_id, face_id in rows:
                image_path = os.path.join(base_path, f"{face_id}.jpg")

                if not os.path.exists(image_path):
                    logger.error(f"Image not found: {image_path} (person_id={person_id})")
                    continue

                try:
                    with open(image_path, "rb") as f:
                        image_bytes = f.read()

                    cur.execute(
                        """
                        UPDATE persons
                        SET thumbnail = %s, updated_at = NOW()
                        WHERE id = %s
                        """,
                        (psycopg2.Binary(image_bytes), person_id)
                    )
                    logger.info(f"Updated thumbnail for person_id={person_id}, face_id={face_id}")

                except Exception as e:
                    logger.error(f"Failed to update person_id={person_id}, face_id={face_id}: {e}")

            conn.commit()
            logger.info(f" Completed processing group {group_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload thumbnails for persons in a group")
    parser.add_argument("group_id", type=int, help="Group ID to process")
    args = parser.parse_args()

    process_group(args.group_id)
    
    with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute("""
                UPDATE groups 
                SET last_processed_step = 'thumbnail_insertion' 
                WHERE id = %s
            """, (args.group_id,))
            conn.commit()
