import psycopg2
from psycopg2.extras import DictCursor

def update_group_image_counts():
    conn = psycopg2.connect(
        host="ballast.proxy.rlwy.net",
        port="56193",
        dbname="railway",
        user="postgres",
        password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
    )
    cur = conn.cursor(cursor_factory=DictCursor)

    try:
        # 1️⃣ Count images per group
        cur.execute("""
            SELECT group_id, COUNT(*) AS img_count
            FROM images
            GROUP BY group_id;
        """)
        results = cur.fetchall()

        if not results:
            print("✅ No images found to update counts.")
            return

        print(f"Found counts for {len(results)} groups.")

        # 2️⃣ Update each group with its image count
        for row in results:
            gid = row["group_id"]
            count = row["img_count"]

            cur.execute(
                "UPDATE groups SET image_count = %s WHERE id = %s;",
                (count, gid)
            )

        conn.commit()
        print("✅ Group image counts updated successfully.")

    except Exception as e:
        conn.rollback()
        print("❌ Error during update:", e)

    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    update_group_image_counts()
