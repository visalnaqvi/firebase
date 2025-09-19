const { Pool } = require('pg');

const pool = new Pool({
    // Pick the DB you want to test
    connectionString: "postgresql://postgres:AfldldzckDWtkskkAMEhMaDXnMqknaPY@ballast.proxy.rlwy.net:56193/railway"
    // connectionString: "postgresql://postgres:kdVrNTrtLzzAaOXzKHaJCzhmoHnSDKDG@nozomi.proxy.rlwy.net:24794/railway"
    // connectionString: "postgresql://postgres:admin@localhost:5432/postgres"
});

async function testInsert() {
    const client = await pool.connect();

    try {
        console.log("🔗 Connected to DB...");

        const query = `
            INSERT INTO images 
            (id, group_id, created_by_user, filename, uploaded_at, status, json_meta_data,
             thumb_byte, image_byte, compressed_location, artist, date_taken, location,
             signed_url, signed_url_3k, signed_url_stripped, last_processed_at)
            VALUES (
                gen_random_uuid(),   -- dummy id
                13,                -- group_id
                '1',                -- created_by_user
                'test',                -- filename
                NULL,                -- uploaded_at
                NULL,                -- status
                NULL,                -- json_meta_data
                NULL,                -- thumb_byte
                NULL,                -- image_byte
                NULL,                -- compressed_location
                NULL,                -- artist
                NULL,                -- date_taken
                NULL,                -- location
                NULL,                -- signed_url
                NULL,                -- signed_url_3k
                NULL,                -- signed_url_stripped
                NOW()                -- last_processed_at (check timezone)
            )
            RETURNING id, last_processed_at;
        `;

        const res = await client.query(query);
        console.log("✅ Inserted row:", res.rows[0]);

    } catch (err) {
        console.error("❌ Error inserting row:", err.message);
    } finally {
        client.release();
        await pool.end();
    }
}

testInsert();
