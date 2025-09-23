import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime, timedelta

def update_groups_to_warm():
    # conn = psycopg2.connect(
    #    host="ballast.proxy.rlwy.net",
    #         port="56193",
    #         dbname="railway",
    #         user="postgres",
    #         password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
    # )
    conn = psycopg2.connect(
        host="nozomi.proxy.rlwy.net",
        port="24794",
        dbname="railway",
        user="postgres",
        password="kdVrNTrtLzzAaOXzKHaJCzhmoHnSDKDG"
    )
    conn.autocommit = True
    cur = conn.cursor(cursor_factory=DictCursor)

    # Step 1: Get all heating groups
    cur.execute("SELECT id FROM groups WHERE status = 'heating'")
    heating_groups = cur.fetchall()

    if not heating_groups:
        print("No heating groups found.")

    for row in heating_groups:
        group_id = row["id"]

        # Step 2: Check if all images meet the condition
        cur.execute("""
            SELECT 
                COUNT(*) FILTER (
                    WHERE status = 'warm' 
                    AND last_processed_at <= (NOW() AT TIME ZONE 'utc') - INTERVAL '1 minutes'
                ) AS valid_images,
                COUNT(*) FILTER (WHERE status = 'warm') AS total_images
            FROM images
            WHERE group_id = %s
        """, (group_id,))
        counts = cur.fetchone()

        valid_images = counts["valid_images"]
        total_images = counts["total_images"]

        if total_images > 0 and valid_images == total_images:
            # Step 3: Update group status to warm
            cur.execute("""
                UPDATE groups
                SET status = 'warm',
                    last_processed_at = NOW()
                WHERE id = %s
            """, (group_id,))
            print(f"[OK] Group {group_id} updated to warm")
        else:
            print(f"[WAIT] Group {group_id} not ready yet ({valid_images}/{total_images} valid images)")

    cur.close()
    conn.close()

if __name__ == "__main__":
    update_groups_to_warm()
