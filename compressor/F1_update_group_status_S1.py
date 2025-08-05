import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta

def get_db_connection():
    return psycopg2.connect(
        host="your_host",
        port="your_port",
        dbname="your_db",
        user="your_user",
        password="your_password"
        # or use os.environ.get("POSTGRES_URL") if using env var
    )

def main():
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        print("✅ Connected to PostgreSQL")

        # Step 1: Get all groups with 'heating' status
        cur.execute("SELECT group_id FROM groups WHERE status = 'heating'")
        heating_groups = [row['group_id'] for row in cur.fetchall()]

        heated_groups = []
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)

        # Step 2–3: Check latest image timestamp for each group
        for group_id in heating_groups:
            cur.execute(
                "SELECT created_at FROM images WHERE group_id = %s ORDER BY created_at DESC LIMIT 1",
                (group_id,)
            )
            row = cur.fetchone()
            if row and row['created_at'] < one_hour_ago:
                heated_groups.append(group_id)

        print(f"🔥 Found {len(heated_groups)} heated groups")

        # Step 4–5: For each heated group, get stats and update
        for group_id in heated_groups:
            cur.execute(
                "SELECT COUNT(*) AS total_images, COALESCE(SUM(size), 0) AS total_size FROM images WHERE group_id = %s",
                (group_id,)
            )
            stats = cur.fetchone()
            total_images = stats['total_images']
            total_size = stats['total_size']

            cur.execute(
                "UPDATE groups SET status = 'hot', total_images = %s, total_size = %s WHERE group_id = %s",
                (total_images, total_size, group_id)
            )

        conn.commit()
        print("✅ Done updating all heated groups")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if conn:
            cur.close()
            conn.close()
            print("🔌 Disconnected from PostgreSQL")

if __name__ == "__main__":
    main()
