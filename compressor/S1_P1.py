import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta

def get_db_connection():
    return psycopg2.connect(
        host="ballast.proxy.rlwy.net",
        port="56193",
        dbname="railway",
        user="postgres",
        password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
        # or use os.environ.get("POSTGRES_URL") if using env var
    )

def main():
    try:
        print("🔃 Connecting to PostgreSQL")
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        print("✅ Connected to PostgreSQL")

        print("🔃 Getting groups with heating status")
        cur.execute("SELECT id FROM groups WHERE status = 'heating'")
        heating_groups = [row['id'] for row in cur.fetchall()]
        print(f"✅ Got {len(heating_groups)} heating groups")
        heated_groups = []
        one_hour_ago = datetime.utcnow() - timedelta(minutes=1)


        for group_id in heating_groups:
            print(f"🔃 Getting the latest image created for group {group_id}")
            cur.execute(
                "SELECT created_at FROM images WHERE group_id = %s ORDER BY created_at DESC LIMIT 1",
                (group_id,)
            )
            row = cur.fetchone()
            print(f"✅ Got the latest image created for group {group_id}")
            if row and row['created_at'] < one_hour_ago:
                print(f"✅ Image was added 1 hour before adding {group_id} to heated groups")
                heated_groups.append(group_id)
            else:
                print(f"❌ Image was added within 1 hour not adding {group_id} to heated groups")
        
        print(f"🔥 Found {len(heated_groups)} heated groups")

        for group_id in heated_groups:
            print(f"🔃 Getting images for {group_id} group")
            cur.execute(
                "SELECT COUNT(*) AS total_images, COALESCE(SUM(size), 0) AS total_size FROM images WHERE group_id = %s",
                (group_id,)
            )
            stats = cur.fetchone()
            total_images = stats['total_images']
            total_size = stats['total_size']
            print(f"✅ Got {total_images} images for {group_id} group with size {total_size}")
            print(f"🔃 Got {total_images} images for {group_id} group with size {total_size}")
            cur.execute(
                "UPDATE groups SET status = 'hot', total_images = %s, total_size = %s WHERE id = %s",
                (total_images, total_size, group_id)
            )
            print(f"✅Updated group status")

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
