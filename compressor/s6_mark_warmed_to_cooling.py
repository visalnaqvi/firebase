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

def mark_groups_as_cooling():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    print("🔄 Checking for eligible groups...")

    # Step 1: Get all groups with status = 'warmed'
    cur.execute("SELECT id FROM groups WHERE status = 'warmed'")
    warmed_groups = [row['id'] for row in cur.fetchall()]

    for group_id in warmed_groups:
        # Step 2: Check if all face records for this group are assigned = true
        cur.execute("""
            SELECT COUNT(*) FILTER (WHERE assigned = TRUE) AS assigned_count,
                   COUNT(*) AS total_count
            FROM faces 
            WHERE group_id = %s
        """, (group_id,))
        result = cur.fetchone()

        if result['assigned_count'] == result['total_count'] and result['total_count'] > 0:
            print(f"✅ Group {group_id} is fully assigned. Marking as cooling...")

            # Update all images of this group to 'cooling'
            cur.execute("""
                UPDATE images SET status = 'cooling'
                WHERE group_id = %s
            """, (group_id,))

            # Update group status to 'cooling'
            cur.execute("""
                UPDATE groups SET status = 'cooling'
                WHERE id = %s
            """, (group_id,))

            conn.commit()
        else:
            print(f"⏳ Group {group_id} still has unassigned faces.")

    cur.close()
    conn.close()
    print("✅ Done.")

if __name__ == "__main__":
    mark_groups_as_cooling()
