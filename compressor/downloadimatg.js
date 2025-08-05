const { Client } = require('pg');
const sharp = require('sharp');
const fs = require('fs');
const path = require('path');
const { Pool } = require('pg');

const pool = new Pool({
    connectionString: "postgresql://postgres:AfldldzckDWtkskkAMEhMaDXnMqknaPY@ballast.proxy.rlwy.net:56193/railway"
});

// Ensure folder exists
const outputDir = path.join(__dirname, 'downloaded-images');
if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
}

async function downloadImages() {
    const client = await pool.connect();

    try {

        console.log('Connected to DB');

        // Select id, filename, image_byte from your images table
        const res = await client.query('SELECT id, filename, image_byte FROM images');

        for (const row of res.rows) {
            const { id, filename, image_byte } = row;

            if (!image_byte) {
                console.warn(`❗ Skipping image ID ${id} - no image_byte found`);
                continue;
            }

            const safeFilename = filename.replace(/[^a-zA-Z0-9.\-_]/g, '_'); // sanitize
            const outputPath = path.join(outputDir, safeFilename);

            try {
                await sharp(image_byte).toFile(outputPath);
                console.log(`✅ Saved: ${outputPath}`);
            } catch (err) {
                console.error(`❌ Error processing image ID ${id}:`, err.message);
            }
        }
    } catch (err) {
        console.error('❌ DB Error:', err);
    } finally {
        client.release();
        console.log('Disconnected from DB');
    }
}

downloadImages();
