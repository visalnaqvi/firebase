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
    connectionString: "postgresql://postgres:your_connection_here"
});

const BATCH_FETCH_SIZE = 100;
const CONCURRENCY = 10;

function getFileNameFromURL(url) {
    return decodeURIComponent(path.basename(new URL(url).pathname));
}

async function fetchHotImages(client, groupId, offset = 0) {
    const res = await client.query(
        `SELECT id, location FROM images WHERE status = 'hot' AND group_id = $1 ORDER BY id LIMIT $2 OFFSET $3`,
        [groupId, BATCH_FETCH_SIZE, offset]
    );
    return res.rows;
}

async function processImage(row) {
    const { id, location } = row;
    const fileName = getFileNameFromURL(location);

    const response = await axios.get(location, { responseType: 'arraybuffer' });
    const originalBuffer = Buffer.from(response.data);
    const originalMeta = await sharp(originalBuffer).metadata();

    const strippedBuffer = await sharp(originalBuffer).withMetadata({}).toBuffer();

    // 👇 Use original image if width < 3000
    const resizedBuffer = originalMeta.width < 3000
        ? strippedBuffer
        : await sharp(strippedBuffer).resize({ width: 3000 }).jpeg().toBuffer();

    const thumbBuffer = await sharp(strippedBuffer).resize({ width: 400 }).jpeg().toBuffer();

    return {
        id,
        json_meta_data: JSON.stringify(originalMeta),
        thumb_byte: thumbBuffer,
        image_byte: resizedBuffer,
    };
}

async function flushBatchUpdates(client, updates) {
    if (!updates.length) return;
    await client.query('BEGIN');
    for (const update of updates) {
        await client.query(
            `UPDATE images SET status = 'warm', json_meta_data = $1, thumb_byte = $2, image_byte = $3 WHERE id = $4`,
            [update.json_meta_data, update.thumb_byte, update.image_byte, update.id]
        );
    }
    await client.query('COMMIT');
}

async function processGroup(client, groupId) {
    console.log(`🔃 Processing group ${groupId}`);
    let offset = 0;
    let updatesBuffer = [];

    while (true) {
        const images = await fetchHotImages(client, groupId, offset);
        if (images.length === 0) break;

        console.log(`✅ Fetched ${images.length} images for group ${groupId} (offset: ${offset})`);

        const chunked = [];
        for (let i = 0; i < images.length; i += CONCURRENCY) {
            chunked.push(images.slice(i, i + CONCURRENCY));
        }

        for (const chunk of chunked) {
            const results = await Promise.allSettled(chunk.map(row => processImage(row)));

            for (const res of results) {
                if (res.status === 'fulfilled') {
                    updatesBuffer.push(res.value);
                } else {
                    console.error('❌ Image processing error:', res.reason.message || res.reason);
                }
            }
        }

        await flushBatchUpdates(client, updatesBuffer);
        updatesBuffer = [];
        offset += BATCH_FETCH_SIZE;
    }

    // ✅ Update group status to warm
    await client.query(`UPDATE groups SET status = 'warm' WHERE id = $1`, [groupId]);
    console.log(`✅ Updated group ${groupId} status to 'warm'`);
}

async function main() {
    const client = await pool.connect();
    try {
        const { rows: groupRows } = await client.query(`SELECT id FROM groups WHERE status = 'hot'`);
        console.log(`🔥 Found ${groupRows.length} hot groups`);

        for (const group of groupRows) {
            await processGroup(client, group.id);
        }
    } catch (err) {
        console.error('❌ Fatal error:', err.message || err);
    } finally {
        await pool.end();
        process.exit();
    }
}

main();
