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

// Configuration
const BATCH_SIZE = 50
const PARALLEL_LIMIT = 10;
const DB_TIMEOUT = 60000;

function getFileNameFromURL(url) {
    return decodeURIComponent(path.basename(new URL(url).pathname));
}

// Process a single image
async function processSingleImage(image) {
    const { id, location, group_id } = image;
    console.log(`🔃 Processing Image ${id} for group ${group_id}`);

    try {
        const fileName = getFileNameFromURL(location);
        console.log(`🔧 Processing image ${fileName} `);

        // Download image
        const response = await axios.get(location, { responseType: 'arraybuffer' });
        const originalBuffer = Buffer.from(response.data);
        console.log(`✅ Downloaded Image ${id} from Firebase`);

        // Get metadata
        const originalMeta = await sharp(originalBuffer).metadata();
        const originalWidth = originalMeta.width;
        console.log(`📏 Image ${id} original width: ${originalWidth} px`);

        // Strip metadata
        const strippedBuffer = await sharp(originalBuffer).withMetadata({}).toBuffer();
        console.log(`✅ Stripped metadata for Image ${id}`);

        // Upload stripped image
        const strippedPath = `stripped / ${fileName} `;
        await bucket.file(strippedPath).save(strippedBuffer, {
            contentType: 'image/jpeg',
        });
        console.log(`✅ Stored stripped Image ${id} to Firebase`);

        // Create 3000px version (or use original if already smaller)
        // let compressedBuffer;
        // if (originalWidth <= 3000) {
        //     // Use stripped buffer as compressed since it's already <= 3000px
        //     compressedBuffer = strippedBuffer;
        //     console.log(`✅ Image ${id} is ${originalWidth}px wide(≤ 3000px), using original size`);
        // } else {
        //     // Resize to 3000px
        //     compressedBuffer = await sharp(strippedBuffer)
        //         .resize({ width: 3000 })
        //         .jpeg()
        //         .toBuffer();
        //     console.log(`✅ Resized Image ${id} from ${originalWidth}px to 3000px`);
        // }

        // const compressedPath = `compressed3 / ${fileName} `;
        // await bucket.file(compressedPath).save(compressedBuffer, {
        //     contentType: 'image/jpeg',
        // });
        // console.log(`✅ Stored 3000px Image ${id} to Firebase`);

        // // Create 400px thumbnail
        // const thumbBuffer = await sharp(strippedBuffer)
        //     .resize({ width: 400 })
        //     .jpeg()
        //     .toBuffer();
        // console.log(`✅ Created 400px thumbnail for Image ${id}`);

        return {
            id,
            success: true,
            data: {
                json_meta_data: JSON.stringify(originalMeta),
                thumb_byte: null,
                image_byte: null,
                status: 'warm'
            }
        };
    } catch (error) {
        console.error(`❌ Error processing Image ${id}: `, error.message);
        return {
            id,
            success: false,
            error: error.message,
            data: {
                json_meta_data: null,
                thumb_byte: null,
                image_byte: null,
                status: 'warm_failed'
            }
        };
    }
}

// Process images in parallel with concurrency limit
async function processImagesBatch(images) {
    console.log(`🔃 Processing batch of ${images.length} images with ${PARALLEL_LIMIT} parallel workers`);

    const results = [];

    // Process images in chunks of PARALLEL_LIMIT
    for (let i = 0; i < images.length; i += PARALLEL_LIMIT) {
        const chunk = images.slice(i, i + PARALLEL_LIMIT);
        console.log(`🔃 Processing chunk ${Math.floor(i / PARALLEL_LIMIT) + 1} /${Math.ceil(images.length / PARALLEL_LIMIT)
            } (${chunk.length} images)`);

        const chunkPromises = chunk.map(image => processSingleImage(image));
        const chunkResults = await Promise.all(chunkPromises);
        results.push(...chunkResults);

        console.log(`✅ Completed chunk ${Math.floor(i / PARALLEL_LIMIT) + 1} /${Math.ceil(images.length / PARALLEL_LIMIT)}`);
    }

    return results;
}

// Attempt single batch update
async function performBatchUpdate(client, successfulResults) {
    const valuesClauses = [];
    const allParams = [];
    let paramIndex = 1;

    for (const result of successfulResults) {
        const { id, data } = result;
        // Fixed: Cast the id parameter properly and ensure correct parameter order
        valuesClauses.push(`($${paramIndex}:: uuid, $${paramIndex + 1}, $${paramIndex + 2}:: jsonb, $${paramIndex + 3}:: bytea, $${paramIndex + 4}:: bytea)`);
        allParams.push(id, data.status, data.json_meta_data, data.thumb_byte, data.image_byte);
        paramIndex += 5;
    }

    const batchUpdateQuery = `
        UPDATE images 
        SET status = data.status,
    json_meta_data = data.json_meta_data,
    thumb_byte = data.thumb_byte,
    image_byte = data.image_byte,
    last_processed_at = NOW()
FROM(VALUES ${valuesClauses.join(', ')}) AS data(id, status, json_meta_data, thumb_byte, image_byte)
        WHERE images.id = data.id
    `;

    console.log(`🔃 Executing batch query with ${allParams.length} parameters...`);
    const startTime = Date.now();

    // Set a timeout for the query (increased to handle large binary data)
    const timeoutPromise = new Promise((_, reject) => {
        setTimeout(() => reject(new Error('Batch update timeout after 60 seconds')), 180000);
    });

    const updateResult = await Promise.race([
        client.query(batchUpdateQuery, allParams),
        timeoutPromise
    ]);

    const duration = Date.now() - startTime;
    console.log(`✅ Successfully batch updated ${updateResult.rowCount} images in database(${duration}ms)`);

    // Verify all records were updated
    if (updateResult.rowCount !== successfulResults.length) {
        console.warn(`⚠️  Expected to update ${successfulResults.length} records, but updated ${updateResult.rowCount} `);
    }
}

// Fallback: chunked updates (smaller batches)
async function performChunkedUpdates(client, successfulResults) {
    const CHUNK_SIZE = 5; // Even smaller chunks for fallback with binary data
    console.log(`🔃 Using chunked updates with ${CHUNK_SIZE} records per chunk`);

    let totalUpdated = 0;
    await client.query('BEGIN');
    try {
        for (let i = 0; i < successfulResults.length; i += CHUNK_SIZE) {
            const chunk = successfulResults.slice(i, i + CHUNK_SIZE);
            console.log(`🔃 Updating chunk ${Math.floor(i / CHUNK_SIZE) + 1} /${Math.ceil(successfulResults.length / CHUNK_SIZE)}`);

            // Process chunk updates sequentially instead of in parallel to reduce load
            for (const result of chunk) {
                const { id, data } = result;
                try {
                    await client.query(
                        `UPDATE images
                         SET status = $1,
    json_meta_data = $2,
    thumb_byte = $3,
    image_byte = $4
                         WHERE id = $5`,
                        [data.status, data.json_meta_data, data.thumb_byte, data.image_byte, id]
                    );
                    totalUpdated++;
                } catch (error) {
                    console.error(`❌ Failed to update image ${id} in chunk: `, error.message);
                }
            }

            console.log(`✅ Updated chunk ${Math.floor(i / CHUNK_SIZE) + 1} /${Math.ceil(successfulResults.length / CHUNK_SIZE)} (${totalUpdated}/${successfulResults.length} total)`);
        }
        await client.query('COMMIT');
    } catch (error) {
        await client.query('ROLLBACK');
        console.log("❌ Chunk update error:", error.message);
    }

    console.log(`✅ Successfully updated ${totalUpdated} images using chunked approach`);
}

// Update database with all results at once using batch UPDATE with fallback
async function updateDatabaseBatch(client, results) {
    const successfulResults = results.filter(r => r.success);
    const failedResults = results.filter(r => !r.success);

    if (failedResults.length > 0) {
        console.log(`⚠️  ${failedResults.length} images failed processing: `,
            failedResults.map(r => `ID: ${r.id}, Error: ${r.error} `));
    }

    if (successfulResults.length === 0) {
        console.log('⚠️  No successful results to update in database');
        return;
    }

    console.log(`🔃 Batch updating database for ${successfulResults.length} successfully processed images`);

    // Calculate total data size for debugging
    let totalDataSize = 0;
    successfulResults.forEach(result => {
        totalDataSize += (result.data.thumb_byte?.length || 0) + (result.data.image_byte?.length || 0);
    });
    console.log(`📊 Total binary data size: ${(totalDataSize / 1024 / 1024).toFixed(2)} MB`);

    // Try batch update first
    try {
        console.log(`🔃 Attempting single batch UPDATE query...`);

        // Begin transaction for batch update
        await client.query('BEGIN');
        await performBatchUpdate(client, successfulResults, "success");
        await performBatchUpdate(client, failedResults, "failure");
        await client.query('COMMIT');

    } catch (batchError) {
        console.log(`⚠️  Batch UPDATE failed: ${batchError.message} `);

        // Rollback the failed batch transaction with timeout
        try {
            console.log(`🔃 Starting Rollback`);

            // Add timeout to rollback operation
            const rollbackPromise = client.query('ROLLBACK');
            const rollbackTimeout = new Promise((_, reject) => {
                setTimeout(() => reject(new Error('Rollback timeout after 10 seconds')), 180000);
            });

            await Promise.race([rollbackPromise, rollbackTimeout]);
            console.log(`✅ Rollback completed`);

        } catch (rollbackError) {
            console.log(`⚠️  Rollback error: ${rollbackError.message} `);

            // If rollback fails, the connection might be in a bad state
            // Let's try to reset the connection state by ending and reconnecting
            console.log(`🔃 Attempting to reset connection state...`);
            try {
                // Try a simple query to test connection
                await client.query('SELECT 1');
                console.log(`✅ Connection still responsive after rollback failure`);
            } catch (connError) {
                console.log(`❌ Connection appears to be in bad state: ${connError.message} `);
                // Note: In a production system, you might want to get a new connection here
            }
        }

        // Try chunked updates in a new transaction
        try {
            console.log(`🔃 Falling back to chunked updates...`);
            await client.query('BEGIN');
            await performChunkedUpdates(client, successfulResults);
            await client.query('COMMIT');

        } catch (chunkError) {
            console.log(`⚠️  Chunk UPDATE failed: ${chunkError.message} `);

            // Rollback the chunked transaction with timeout
            try {
                console.log(`🔃 Starting chunked update rollback`);

                const rollbackPromise = client.query('ROLLBACK');
                const rollbackTimeout = new Promise((_, reject) => {
                    setTimeout(() => reject(new Error('Chunked rollback timeout after 10 seconds')), 180000);
                });

                await Promise.race([rollbackPromise, rollbackTimeout]);
                console.log(`✅ Chunked update rollback completed`);

            } catch (rollbackError) {
                console.log(`⚠️  Chunk rollback error: ${rollbackError.message} `);
            }

            // Try individual updates as last resort
            console.log(`🔃 Falling back to individual updates...`);
            let individualSuccessCount = 0;

            for (const result of successfulResults) {
                try {
                    await client.query('BEGIN');
                    const { id, data } = result;
                    const updateResult = await client.query(
                        `UPDATE images
                         SET status = $1,
    json_meta_data = $2,
    thumb_byte = $3,
    image_byte = $4
                         WHERE id = $5`,
                        [data.status, data.json_meta_data, data.thumb_byte, data.image_byte, id]
                    );
                    await client.query('COMMIT');

                    if (updateResult.rowCount > 0) {
                        individualSuccessCount++;
                    }
                } catch (individualError) {
                    console.error(`❌ Failed to update image ${result.id}: `, individualError.message);
                    try {
                        const rollbackPromise = client.query('ROLLBACK');
                        const rollbackTimeout = new Promise((_, reject) => {
                            setTimeout(() => reject(new Error('Individual rollback timeout after 5 seconds')), 180000);
                        });
                        await Promise.race([rollbackPromise, rollbackTimeout]);
                    } catch (rollbackError) {
                        console.log(`⚠️  Individual rollback error: ${rollbackError.message} `);
                    }
                }
            }

            console.log(`✅ Successfully updated ${individualSuccessCount}/${successfulResults.length} images using individual updates`);
        }
    }
}

// Fetch all hot images for a specific group
async function fetchAllHotImages(client, groupId) {
    const { rows } = await client.query(
        `SELECT id, location, group_id FROM images 
         WHERE status = 'hot' AND group_id = $1 
         ORDER BY id limit 100`,
        [groupId]
    );
    return rows;
}

// Process images in batches of BATCH_SIZE with database update after each batch
async function processImagesBatches(client, images, groupId) {
    console.log(`🔃 Processing ${images.length} images in batches of ${BATCH_SIZE} for group ${groupId}`);

    let totalProcessed = 0;
    let totalFailed = 0;

    // Process images in batches of BATCH_SIZE
    for (let i = 0; i < images.length; i += BATCH_SIZE) {
        const batch = images.slice(i, i + BATCH_SIZE);
        const batchNumber = Math.floor(i / BATCH_SIZE) + 1;
        const totalBatches = Math.ceil(images.length / BATCH_SIZE);

        console.log(`🔃 Processing batch ${batchNumber}/${totalBatches} (${batch.length} images) for group ${groupId}`);

        // Process this batch
        const batchResults = await processImagesBatch(batch);

        // Update database immediately after processing this batch
        console.log(`🔃 Updating database for batch ${batchNumber}/${totalBatches}`);
        await updateDatabaseBatch(client, batchResults);

        const successfulCount = batchResults.filter(r => r.success).length;
        const failedCount = batchResults.filter(r => !r.success).length;

        totalProcessed += successfulCount;
        totalFailed += failedCount;

        console.log(`✅ Completed batch ${batchNumber}/${totalBatches}. Batch: ${successfulCount}/${batch.length} successful, ${failedCount} failed. Total: ${totalProcessed} processed, ${totalFailed} failed`);

        // Optional: Add a small delay between batches to prevent overwhelming the system
        if (batchNumber < totalBatches) {
            console.log(`⏸️  Brief pause before next batch...`);
            await new Promise(resolve => setTimeout(resolve, 100)); // 100ms pause
        }
    }

    return { totalProcessed, totalFailed };
}

// Process a single group with batching
async function processGroup(client, groupId) {
    console.log(`🔃 Starting processing for group ${groupId}`);
    let totalProcessedFinal = []
    let totalFailedFinal = []
    while (true) {
        console.log(`🔃 Fetching all hot images for group ${groupId}`);
        const allImages = await fetchAllHotImages(client, groupId);

        if (allImages.length === 0) {
            console.log(`✅ No hot images found for group ${groupId}`);
            break;
        }

        console.log(`✅ Fetched ${allImages.length} hot images for group ${groupId}`);

        // Process all images in batches with database updates after each batch
        const { totalProcessed, totalFailed } = await processImagesBatches(client, allImages, groupId);
        totalProcessedFinal.push(totalProcessed)
        totalFailedFinal.push(totalFailed)
        console.log(`✅ Completed processing for group ${groupId}. Total: ${totalProcessed}/${allImages.length} processed successfully, ${totalFailed} failed`);
    }
    // Fetch ALL hot images for this group at once
    return { totalProcessedFinal, totalFailedFinal };
}

async function processImages() {
    const client = await pool.connect();
    try {
        while (true) {
            console.log("🔃 Getting hot groups to process images");
            const { rows: groupRows } = await client.query(`SELECT id FROM groups WHERE status = 'heating' and last_image_uploaded_at is not null order by last_image_uploaded_at LIMIT 1`);
            console.log(`✅ Found ${groupRows.length} hot groups`);

            let totalProcessedAllGroups = 0;
            if (groupRows.length == 0) {
                console.log("No Groups to process exiting")
                break;
            }
            for (const group of groupRows) {
                const processedCount = await processGroup(client, group.id);
                totalProcessedAllGroups += processedCount;
            }

            console.log(`🎉 Processing complete! Total images processed: ${totalProcessedAllGroups}`);
        }


    } catch (err) {
        console.error('❌ Error processing images:', err);
    } finally {
        client.release();
        await pool.end();
        process.exit();
    }
}
processImages();