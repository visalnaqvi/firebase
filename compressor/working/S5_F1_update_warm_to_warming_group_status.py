import psycopg2
from psycopg2.extras import RealDictCursor

def get_db_connection():
    return psycopg2.connect(
        host="localhost",
        port="5432",
        dbname="postgres",
        user="postgres",
        password="admin"
    )

def main():
    conn = None
    try:
        print("🔃 Connecting to PostgreSQL")
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        print("✅ Connected to PostgreSQL")

        print("🔃 Updating 'warm' groups to 'warming' if they meet both conditions")
        update_sql = """
        UPDATE groups g
        SET status = 'warming'
        WHERE g.status = 'warm'
          -- Condition 1: No images with 'hot' or 'warm'
          AND NOT EXISTS (
              SELECT 1 FROM images i
              WHERE i.group_id = g.id
                AND i.status IN ('hot', 'warm')
          )
          -- Condition 2: At least one image with 'warming'
          AND EXISTS (
              SELECT 1 FROM images i
              WHERE i.group_id = g.id
                AND i.status = 'warming'
          );
        """
        cur.execute(update_sql)
        affected_rows = cur.rowcount
        conn.commit()

        print(f"✅ Updated {affected_rows} groups to 'warming'")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if conn:
            cur.close()
            conn.close()
            print("🔌 Disconnected from PostgreSQL")

if __name__ == "__main__":
    main()
