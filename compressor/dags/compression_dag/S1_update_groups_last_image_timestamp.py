import psycopg2
from psycopg2.extras import DictCursor
import logging
import time
# --- Configure logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def update_groups_metadata():
    try:
        conn = psycopg2.connect(
            host="ballast.proxy.rlwy.net",
            port="56193",
            dbname="railway",
            user="postgres",
            password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
        )
        cur = conn.cursor(cursor_factory=DictCursor)

        # 1. Get all groups with heating status
        cur.execute("SELECT id FROM groups")
        groups = cur.fetchall()

        if not groups:
            logging.info("No groups found with status = 'heating'")
            return

        for group in groups:
            group_id = group["id"]

            # 2. Get most recent image timestamp, total images, and total size
            cur.execute("""
                SELECT 
                    MAX(uploaded_at) AS latest_uploaded_at,
                    COUNT(*) AS total_images,
                    COALESCE(SUM(size), 0) AS total_size
                FROM images
                WHERE group_id = %s
            """, (group_id,))
            row = cur.fetchone()

            latest_uploaded_at = row["latest_uploaded_at"]
            total_images = row["total_images"]
            total_size = row["total_size"]

            if total_images > 0:
                # 3. Update groups table with aggregated values
                cur.execute("""
                    UPDATE groups
                    SET last_image_uploaded_at = %s,
                        total_images = %s,
                        total_size = %s
                    WHERE id = %s
                """, (latest_uploaded_at, total_images, total_size, group_id))

                logging.info(
                    f"Updated group {group_id} → "
                    f"last_image_uploaded_at={latest_uploaded_at}, "
                    f"total_images={total_images}, total_size={total_size}"
                )
            else:
                logging.info(f"No images found for group {group_id}")

        conn.commit()
        cur.close()
        conn.close()
        logging.info("All updates committed successfully.")

    except Exception as e:
        logging.error(f"Error updating groups: {e}", exc_info=True)


if __name__ == "__main__":
    update_groups_metadata()
    print("Sleeping for 5 minutes...")
    time.sleep(300)  # 300 seconds = 5 minutes
