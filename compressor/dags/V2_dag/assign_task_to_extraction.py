import psycopg2
from psycopg2.extras import DictCursor
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def update_process_queue():
    try:
        conn = psycopg2.connect(
            host="ballast.proxy.rlwy.net",
            port="56193",
            dbname="railway",
            user="postgres",
            password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
        )
        cur = conn.cursor(cursor_factory=DictCursor)

        # 1. Fetch two oldest warm groups
        cur.execute("""
            SELECT id
            FROM groups
            WHERE status = 'warm'
            ORDER BY last_image_uploaded_at ASC
            LIMIT 2
        """)
        warm_groups = [row["id"] for row in cur.fetchall()]

        if not warm_groups:
            logging.info("No warm groups found, nothing to update.")
            return

        group1 = warm_groups[0] if len(warm_groups) > 0 else None
        group2 = warm_groups[1] if len(warm_groups) > 1 else None

        # 2. Get the current process_status for extraction
        cur.execute("""
            SELECT id, processing_group, next_group_in_queue, task_status
            FROM process_status
            WHERE task = 'extraction' AND task_status != 'failed'
            LIMIT 1
        """)
        row = cur.fetchone()
        if not row:
            logging.info("No extraction task found in process_status.")
            return

        process_id = row["id"]
        processing_group = row["processing_group"]
        next_group = row["next_group_in_queue"]

        # 3. Apply the rules
        if next_group and processing_group is None:
            # Case 1: next_group exists, processing_group is null → move next → processing
            cur.execute("""
                UPDATE process_status
                SET processing_group = next_group_in_queue,
                    next_group_in_queue = NULL
                WHERE id = %s
            """, (process_id,))
            logging.info(f"Moved next_group_in_queue ({next_group}) → processing_group")

        elif processing_group and next_group is None and group1:
            # Case 2: processing_group exists, next_group empty → fill with oldest warm group
            cur.execute("""
                UPDATE process_status
                SET next_group_in_queue = %s
                WHERE id = %s
            """, (group1, process_id))
            logging.info(f"Set next_group_in_queue = {group1}")

        elif processing_group is None and next_group is None:
            # Case 3: both empty → assign group1 to processing, group2 to next
            cur.execute("""
                UPDATE process_status
                SET processing_group = %s,
                    next_group_in_queue = %s
                WHERE id = %s
            """, (group1, group2, process_id))
            logging.info(f"Set processing_group = {group1}, next_group_in_queue = {group2}")

        else:
            logging.info("No updates needed for process_status.")

        conn.commit()
        cur.close()
        conn.close()
        logging.info("Process queue update committed successfully.")

    except Exception as e:
        logging.error(f"Error updating process queue: {e}", exc_info=True)


if __name__ == "__main__":
    update_process_queue()
