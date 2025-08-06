import psycopg2
from psycopg2.extras import RealDictCursor

def get_db_connection():
    return psycopg2.connect(
        host="your_host",
        port="your_port",
        dbname="your_db",
        user="your_user",
        password="your_password"
    )

def main():
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        print("✅ Connected to PostgreSQL")

        print("🔃 Getting all groups with status 'warm'")
        cur.execute("SELECT id FROM groups WHERE status = 'warm'")
        warm_groups = [row['id'] for row in cur.fetchall()]
        print(f"✅ Got {len(warm_groups)} warm groups")
        updated_groups = 0

        for group_id in warm_groups:
            print(f"🔃 Processing group {group_id}")
            print(f"🔃 Getting Image for {group_id}")
            cur.execute("SELECT count(*) FROM images WHERE group_id = %s and status = %s", (group_id,"warmed"))
            result = cur.fetchone()
            count = result['count']
            if count > 0:
                cur.execute("UPDATE groups SET status = %s where id = %s", ("hot" , group_id))
                print(f"✅ Updated Status for group {group_id} as hot")
            else:
                cur.execute("UPDATE groups SET status = %s where id = %s", ("warmed" , group_id))
                updated_groups += 1
                print(f"✅ Updated Status for group {group_id} as warmed")

        conn.commit()
        print(f"✅ Done. Total groups updated to 'warmed': {updated_groups}")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if conn:
            cur.close()
            conn.close()
            print("🔌 Disconnected from PostgreSQL")

if __name__ == "__main__":
    main()
