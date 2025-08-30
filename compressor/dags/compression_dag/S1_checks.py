import psycopg2

def get_db_connection():
    return psycopg2.connect(
         host="ballast.proxy.rlwy.net",
        port="56193",
        dbname="railway",
        user="postgres",
        password="AfldldzckDWtkskkAMEhMaDXnMqknaPY"
    )

TABLES = [
    """
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    """,
    """
    CREATE TABLE IF NOT EXISTS public.faces (
        id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
        face_emb bytea NULL,
        clothing_emb bytea NULL,
        assigned bool NULL,
        image_id uuid NOT NULL,
        group_id int NOT NULL,
        person_id uuid NULL,
        cropped_img_byte bytea NULL,
        face_thumb_bytes bytea NULL,
        created_at timestamp NULL,
        quality_score float8 NULL,
        insight_face_confidence float8
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS public."groups" (
        id SERIAL PRIMARY KEY,
        "name" text NOT NULL,
        "profile_pic_location" text NULL,
        status text NOT NULL,
        total_images int4 NULL,
        total_size int8 NULL,
        admin_user int NOT NULL,
        last_processed_at timestamp NULL,
        created_at timestamp NULL,
        last_image_uploaded_at timestamp NULL,
        last_processed_step text,
        plan_type text,
        access text,
        profile_pic_bytes bytea,
        delete_at timestamp
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS public.images (
        id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
        group_id int NOT NULL,
        filename text NOT NULL,
        "location" text NULL,
        compressed_location text,
        delete_at timestamp NULL,
        status text NULL,
        json_meta_data jsonb NULL,
        thumb_byte bytea NULL,
        image_byte bytea NULL,
        uploaded_at timestamp NULL,
        last_accessed_at timestamp NULL,
        last_downloaded_at timestamp NULL,
        created_by_user int NOT NULL,
        last_processed_at timestamp Null,
        "size" int8 NULL,
        artist text,
        date_created text,
        signed_url TEXT,
        expire_time TIMESTAMP,
        date_taken TIMESTAMP,
        highlight boolean
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS public.users (
        id SERIAL PRIMARY KEY,
        first_name text NOT NULL,
        last_name text NOT NULL,
        is_admin bool NULL,
        "groups" int[] NULL,
        date_of_birth date NULL,
        profile_pic_bytes bytea NULL,
        person_id int NULL,
        email text NULL,
        phone_number text NULL,
        password_hash text NULL,
        created_at timestamp NULL,
        plan_type text
    );
    """,
    """
    CREATE TABLE persons (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(), 
    thumbnail BYTEA,                               
    name TEXT,                                     
    user_id INT,                                   
    image_ids UUID[] ,
	group_id int,
    total_images int,
 face_id uuid 
);
""",
    """
        CREATE TABLE IF NOT EXISTS similar_faces (
            id SERIAL PRIMARY KEY,
            group_id VARCHAR(255),
            person_id VARCHAR(255),
            similar_person_id VARCHAR(255)
        );
    """,
    
    # Trigger function to send notification
    """
    CREATE TABLE process_status (
    group_id INT,
    status TEXT
);
    """,
    """
    INSERT INTO process_status (group_id, status)
VALUES (NULL, 'extraction');
    """
]

def ensure_tables_exist():
    conn = get_db_connection()
    cur = conn.cursor()

    for create_sql in TABLES:
        cur.execute(create_sql)

    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    try:
        ensure_tables_exist()
        print("✅ Tables and trigger created successfully")
    except Exception as e:
        print(f"❌ Something went wrong: {str(e)}")
