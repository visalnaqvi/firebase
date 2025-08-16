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
        person_id int NULL,
        cropped_img_byte bytea NULL,
        face_thumb_bytes bytea NULL,
        created_at timestamp NULL,
        quality_score float8 NULL
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS public."groups" (
        id SERIAL PRIMARY KEY,
        "name" text NOT NULL,
        status text NOT NULL,
        total_images int4 NULL,
        total_size int8 NULL,
        admin_user int NOT NULL,
        last_processed_at timestamp NULL,
        created_at timestamp NULL,
        last_image_uploaded_at timestamp NULL,
        last_processed_step text
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS public.images (
        id uuid PRIMARY KEY DEFAULT uuid_generate_v4(),
        group_id int NOT NULL,
        filename text NOT NULL,
        "location" text NOT NULL,
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
        "size" int8 NULL
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
        created_at timestamp NULL
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
    CREATE OR REPLACE FUNCTION notify_group_status_change()
    RETURNS trigger AS $$
    BEGIN
        RAISE NOTICE 'Trigger fired: OLD.status=%, NEW.status=%', OLD.status, NEW.status;
        IF NEW.status = 'heating' AND OLD.status IS DISTINCT FROM NEW.status THEN
            PERFORM pg_notify('group_status_channel', NEW.id::text);
        END IF;
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
    """,
    # Trigger itself
    """
    DROP TRIGGER IF EXISTS group_status_trigger ON public."groups";
    CREATE TRIGGER group_status_trigger
    AFTER UPDATE OF status ON public."groups"
    FOR EACH ROW
    EXECUTE FUNCTION notify_group_status_change();
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
