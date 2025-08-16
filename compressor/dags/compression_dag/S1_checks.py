import psycopg2
from psycopg2.extras import DictCursor
import logging

# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# --- DB Connection Settings ---
DB_CONFIG = {
    "dbname": "railway",
    "user": "your_user",
    "password": "your_password",
    "host": "your_host",
    "port": 5432,
}

def update_groups_last_image_timestamp():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor(cursor_factory=DictCursor)

        # 1. Get all groups with heating status
        cur.execute("SELECT id FROM groups WHERE status = 'heating'")
        groups = cur.fetchall()

        if not groups:
            logging.info("No groups found with status = 'heating'")
            return

        for group in groups:
            group_id = group["id"]

            # 2. Get most recent image for this group
            cur.execute("""
                SELECT uploaded_at
                FROM images
                WHERE group_id = %s
                ORDER BY uploaded_at DESC
                LIMIT 1
            """, (group_id,))
            row = cur.fetchone()

            if row and row["uploaded_at"]:
                latest_uploaded_at = row["uploaded_at"]

                # 3. Update groups.last_image_uploaded_at
                cur.execute("""
                    UPDATE groups
                    SET last_image_uploaded_at = %s
                    WHERE id = %s
                """, (latest_uploaded_at, group_id))

                logging.info(f"Updated group {group_id} with last_image_uploaded_at = {latest_uploaded_at}")
            else:
                logging.info(f"No images found for group {group_id}")

        conn.commit()
        cur.close()
        conn.close()
        logging.info("All updates committed successfully.")

    except Exception as e:
        logging.error(f"Error updating groups: {e}", exc_info=True)


if __name__ == "__main__":
    update_groups_last_image_timestamp()
