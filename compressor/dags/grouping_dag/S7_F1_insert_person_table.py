import psycopg2

def get_db_connection():
    return psycopg2.connect(
        host="ballast.proxy.rlwy.net",
        port="56193",
        dbname="railway",
        user="postgres",
        password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
    )

def sync_persons():
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # 1️⃣ Get all warmed groups
        cur.execute("SELECT id FROM groups WHERE status = 'warmed'")
        group_ids = [row[0] for row in cur.fetchall()]

        print(f"Found warmed groups: {group_ids}")

        for group_id in group_ids:
            print(f"Processing group {group_id}")

            # 2️⃣ Insert or update persons (with total_images instead of image_ids)
            cur.execute(
                """
                INSERT INTO persons (id, thumbnail, name, user_id, image_ids, group_id, quality_score, face_id, total_images)
                WITH best_faces AS (
                    SELECT DISTINCT ON (person_id)
                        person_id,
                        face_thumb_bytes,
                        quality_score,
                        id
                    FROM faces
                    WHERE group_id = %s
                      AND person_id IS NOT NULL
                      AND face_thumb_bytes IS NOT NULL
                    ORDER BY person_id, quality_score DESC NULLS LAST
                ),
                image_counts AS (
                    SELECT person_id, COUNT(*) AS total_images
                    FROM faces
                    WHERE group_id = %s
                      AND person_id IS NOT NULL
                    GROUP BY person_id
                )
                SELECT 
                    b.person_id,
                    b.face_thumb_bytes,
                    NULL,   -- name placeholder
                    NULL,   -- user_id placeholder
                    NULL,   -- image_ids now always NULL
                    %s as group_id,
                    b.quality_score,
                    b.id as face_id,
                    c.total_images
                FROM best_faces b
                JOIN image_counts c ON b.person_id = c.person_id
                ON CONFLICT (id) DO UPDATE
                SET total_images = EXCLUDED.total_images,
                    group_id = EXCLUDED.group_id,
                    quality_score = EXCLUDED.quality_score
                """,
                (group_id, group_id, group_id)
            )

        conn.commit()
        print("✅ Sync complete")

    except Exception as e:
        conn.rollback()
        print("❌ Error:", e)

    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    sync_persons()
