const { Pool } = require('pg');
const admin = require('firebase-admin');
const sharp = require('sharp');
const axios = require('axios');
const path = require('path');
const { URL } = require('url');
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
        console.log("🔃 Getting hot groups to stripe and resize images")
        const { rows: groupRows } = await client.query(`SELECT id FROM groups WHERE status = 'hot'`);
        console.log("✅ Got " + groupRows.length + " Hot Groups")
        for (const group of groupRows) {
            const g_id = group.id;
            console.log("🔃 Processing group " + g_id + " fetching all hot images with there id and location")
            const { rows } = await client.query(`SELECT id, location FROM images WHERE status = 'hot' AND group_id = $1`, [g_id]);
            console.log("✅ Got " + rows.length + " Hot Images for group " + g_id)
            for (const row of rows) {
                const { id, location } = row;
                console.log("🔃 Processing Image " + id + " for group " + g_id)
                const fileName = getFileNameFromURL(location);

                console.log(`🔧 Processing image ${fileName}`);


                const response = await axios.get(location, { responseType: 'arraybuffer' });
                const originalBuffer = Buffer.from(response.data);

                console.log("Got Image from Firebase now procssing it")
                const originalMeta = await sharp(originalBuffer).metadata();


                const strippedBuffer = await sharp(originalBuffer).withMetadata({}).toBuffer();

                console.log("🔃 Stripped the Image " + id + " of meta data , trying to store back to firebase")
                const strippedPath = `stripped/${fileName}`;
                await bucket.file(strippedPath).save(strippedBuffer, {
                    contentType: 'image/jpeg',
                });

                console.log("✅ Stored Stripped Image" + id + " back to firebase ")
                console.log("🔃 Resizing Image " + id + " to 3000 pixels")

                const compressedBuffer = await sharp(strippedBuffer)
                    .resize({ width: 3000 })
                    .jpeg()
                    .toBuffer();
                const compressedPath = `compressed3/${fileName}`;
                console.log("🔃 Saving 3000px Image " + id + " to Firebase")
                await bucket.file(compressedPath).save(compressedBuffer, {
                    contentType: 'image/jpeg',
                });
                console.log("✅ Stored 3000px Image" + id + " back to firebase ")
                console.log("🔃 Resizing Image " + id + " to 400 pixels")
                const thumbBuffer = await sharp(strippedBuffer)
                    .resize({ width: 400 })
                    .jpeg()
                    .toBuffer();

                console.log("🔃 Updating Database for Image " + id + " adding status as warm , json_meta data , thumb_byte , image_byte")
                await client.query(
                    `UPDATE images
                     SET status = 'warm',
                         json_meta_data = $1,
                         thumb_byte = $2,
                         image_byte = $3
                     WHERE id = $4`,
                    [JSON.stringify(originalMeta), thumbBuffer, compressedBuffer, id]
                );
                console.log("✅ Updated Database for Image" + id)
            }

            console.log(`🔃 Updating group status: ${g_id}`)
            await client.query(
                `UPDATE groups
                 SET status = 'warm'
                 WHERE id = $1`,
                [g_id]
            );

            console.log(`✅ Updated group status: ${g_id}`);
        }
    } catch (err) {
        console.error('❌ Error processing images:', err);
    } finally {
        client.release();
        process.exit();
    }
}

processImages();
