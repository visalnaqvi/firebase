import psycopg2
from psycopg2.extras import RealDictCursor

def get_db_connection():
    # return psycopg2.connect(
    #     host="ballast.proxy.rlwy.net",
    #     port="56193",
    #     dbname="railway",
    #     user="postgres",
    #     password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
    # )
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

        print("🔃 Updating 'hot' groups with no 'hot' images to 'warm'")
        update_sql = """
        UPDATE groups g
        SET status = 'warm'
        WHERE g.status = 'hot'
          AND NOT EXISTS (
              SELECT 1
              FROM images i
              WHERE i.group_id = g.id
                AND i.status = 'hot'
          );
        """
        cur.execute(update_sql)
        affected_rows = cur.rowcount
        conn.commit()

        print(f"✅ Updated {affected_rows} groups to 'warm'")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if conn:
            cur.close()
            conn.close()
            print("🔌 Disconnected from PostgreSQL")

if __name__ == "__main__":
    main()
