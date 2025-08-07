import psycopg2
from psycopg2.extras import RealDictCursor

def get_db_connection():
    return psycopg2.connect(
        host="ballast.proxy.rlwy.net",
        port="56193",
        dbname="railway",
        user="postgres",
        password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
    )

def main():
    conn = None
    try:
        print("🔃 Connecting to PostgreSQL")
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        print("✅ Connected to PostgreSQL")

        print("🔃 Getting groups with hot status")
        cur.execute("SELECT id FROM groups WHERE status = 'hot'")
        hot_groups = [row['id'] for row in cur.fetchall()]
        print(f"✅ Got {len(hot_groups)} hot groups")

        for group_id in hot_groups:
            print(f"🔃 Getting images for group {group_id}")
            cur.execute(
                "SELECT COUNT(*) AS total_images FROM images WHERE group_id = %s AND status = 'hot'",
                (group_id,)
            )
            stats = cur.fetchone()
            total_images = stats['total_images']

            if total_images > 0:
                print(f"Skipping group {group_id}")
            else:
                cur.execute(
                    "UPDATE groups SET status = 'warm' WHERE id = %s",
                    (group_id,)
                )
                print(f"✅ Updated group {group_id} status to warm")

        conn.commit()
        print("✅ Done updating all groups")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if conn:
            cur.close()
            conn.close()
            print("🔌 Disconnected from PostgreSQL")

if __name__ == "__main__":
    main()
