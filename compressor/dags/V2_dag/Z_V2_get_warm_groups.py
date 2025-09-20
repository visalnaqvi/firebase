import psycopg2

def get_db_connection():
    return psycopg2.connect(
        host="ballast.proxy.rlwy.net",
        port="56193",
        dbname="railway",
        user="postgres",
        password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
    )

def get_warm_groups():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id
                FROM groups
                WHERE status = 'warm'
                ORDER BY last_image_uploaded_at ASC
            """)
            rows = cur.fetchall()

            if not rows:
                return []

            # Return only the ids
            return [str(r[0]) for r in rows]

if __name__ == "__main__":
    groups = get_warm_groups()
    if groups:
        # Print space-separated ids so shell can capture them
        print(" ".join(groups))
    else:
        print("")  # nothing found
