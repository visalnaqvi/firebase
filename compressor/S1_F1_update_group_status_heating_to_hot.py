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
    try:
        print("🔃 Connecting to PostgreSQL")
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        print("✅ Connected to PostgreSQL")

        print("🔃 Updating heated groups in one query")
        update_sql = """
        WITH heated_groups AS (
            SELECT g.id AS group_id,
                   COUNT(i.id) AS total_images,
                   COALESCE(SUM(i.size), 0) AS total_size
            FROM groups g
            JOIN images i ON i.group_id = g.id
            WHERE g.status = 'heating'
            GROUP BY g.id
            HAVING MAX(i.created_at) < NOW() - INTERVAL '1 hour'
        )
        UPDATE groups g
        SET status = 'hot',
            total_images = h.total_images,
            total_size = h.total_size
        FROM heated_groups h
        WHERE g.id = h.group_id;
        """

        cur.execute(update_sql)
        affected_rows = cur.rowcount
        conn.commit()

        print(f"✅ Updated {affected_rows} groups to 'hot' status")

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        if conn:
            cur.close()
            conn.close()
            print("🔌 Disconnected from PostgreSQL")

if __name__ == "__main__":
    main()
