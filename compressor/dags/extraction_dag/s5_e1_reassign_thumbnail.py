import psycopg2

def get_db_connection():
    return psycopg2.connect(
        host="ballast.proxy.rlwy.net",
        port="56193",
        dbname="railway",
        user="postgres",
        password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
    )

def update_persons_with_best_face():
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # 🔹 Update persons with highest quality face for each person_id
        cur.execute(
            """
            WITH best_faces AS (
                SELECT DISTINCT ON (person_id)
                    person_id,
                    id as face_id,
                    quality_score,
                    face_thumb_bytes
                FROM faces
                WHERE person_id IS NOT NULL
                ORDER BY person_id, quality_score DESC NULLS LAST
            )
            UPDATE persons p
            SET quality_score = bf.quality_score,
                thumbnail = bf.face_thumb_bytes
            FROM best_faces bf
            WHERE p.id = bf.person_id;
            """
        )

        conn.commit()
        print("✅ Persons table updated with best faces")

    except Exception as e:
        conn.rollback()
        print("❌ Error:", e)

    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    update_persons_with_best_face()
