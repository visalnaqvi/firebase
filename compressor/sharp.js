const { Pool } = require('pg');
const admin = require('firebase-admin');
const sharp = require('sharp');
const axios = require('axios');
const path = require('path');


const serviceAccount = require('./firebase-key.json');

admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
    storageBucket: 'gallery-585ee.firebasestorage.app',
});
const bucket = admin.storage().bucket();

const pool = new Pool({
    connectionString: "postgresql://postgres:AfldldzckDWtkskkAMEhMaDXnMqknaPY@ballast.proxy.rlwy.net:56193/railway"
});

function getFileNameFromURL(url) {
    return decodeURIComponent(path.basename(new URL(url).pathname));
}

async function processImages() {
    const client = await pool.connect();
    try {
        const { rows } = await client.query(`SELECT id, location FROM images WHERE status = 'warm'`);

        for (const row of rows) {
            const { id, location } = row;
            const fileName = getFileNameFromURL(location);

            console.log(`🔧 Processing image ${fileName}`);

            // Step 1: Download image
            const response = await axios.get(location, { responseType: 'arraybuffer' });
            const originalBuffer = Buffer.from(response.data);

            // Step 2: Extract original metadata
            const originalMeta = await sharp(originalBuffer).metadata();

            // Step 3: Strip metadata
            const strippedBuffer = await sharp(originalBuffer).withMetadata({}).toBuffer();

            // Step 4: Upload stripped image to /stripped/
            const strippedPath = `stripped/${fileName}`;
            await bucket.file(strippedPath).save(strippedBuffer, {
                contentType: 'image/jpeg',
            });

            // Step 5: Generate 2000px compressed image
            const compressedBuffer = await sharp(strippedBuffer)
                .resize({ width: 3000 })
                .jpeg()
                .toBuffer();

            const compressedPath = `compressed3/${fileName}`;
            await bucket.file(compressedPath).save(compressedBuffer, {
                contentType: 'image/jpeg',
            });

            // Step 6: Generate thumbnail (400px width)
            const thumbBuffer = await sharp(strippedBuffer).resize({ width: 400 }).jpeg().toBuffer();

            // Step 7: Update DB
            await client.query(
                `UPDATE images
             SET status = 'warm',
                 json_meta_data = $1,
                 thumb_byte = $2,
                 image_byte = $3
             WHERE id = $4`,
                [JSON.stringify(originalMeta), thumbBuffer, compressedBuffer, id]
            );

            console.log(`✅ Updated and uploaded: ${fileName}`);
        }
    } catch (err) {
        console.error('❌ Error processing images:', err);
    } finally {
        client.release();
        process.exit();
    }
}

processImages();
