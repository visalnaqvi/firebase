const { Pool } = require('pg');
const admin = require('firebase-admin');
const sharp = require('sharp');
const path = require('path');
const serviceAccount = require('./firebase-key.json');
const exifParser = require('exif-parser');
const { error } = require('console');
const fs = require("fs").promises

admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
    // DEV BUCKET
    storageBucket: 'gallery-585ee.firebasestorage.app',
    // PROD BUCKET
    // storageBucket: 'gallery-585ee-production',
});
const bucket = admin.storage().bucket();

const pool = new Pool({
    // PROD DATABASE
    // connectionString: "postgresql://postgres:kdVrNTrtLzzAaOXzKHaJCzhmoHnSDKDG@nozomi.proxy.rlwy.net:24794/railway"
    // DEV DATABASE
    connectionString: "postgresql://postgres:AfldldzckDWtkskkAMEhMaDXnMqknaPY@ballast.proxy.rlwy.net:56193/railway"
    // connectionString: "postgresql://postgres:admin@localhost:5432/postgres"
});
class ProcessingError extends Error {
    constructor(message, { groupId = null, reason = null, retryable = true } = {}) {
        super(message);
        this.name = "ProcessingError";
        this.groupId = groupId;
        this.reason = reason;
        this.retryable = retryable;
    }
}

// Configuration
const BATCH_SIZE = 10
const PARALLEL_LIMIT = 10;
const DB_TIMEOUT = 60000;
const FIREBASE_BATCH_SIZE = 50; // Number of files to fetch from Firebase at once
const CLEANUP_BATCH_SIZE = 20; // Number of files to cleanup in parallel
const MAX_CLEANUP_RETRIES = 3; // Maximum retries for cleanup operations


async function getDynamicParallelLimit(client) {
    try {
        const { rows } = await client.query(
            "SELECT running FROM process_status WHERE process = $1",
            ['extraction']
        );
        if (rows.length && rows[0].running === true) {
            console.log("Extraction running, using PARALLEL_LIMIT=2");
            return 2;
        }
    } catch (error) {
        console.warn("Could not check process_status, using default limit (10). Error:", error.message);
    }
    return 10;
}

// Fetch a batch of unprocessed images from Firebase Storage with pagination
async function fetchUnprocessedImagesBatch(maxResults = FIREBASE_BATCH_SIZE, pageToken = null) {
    console.log(`🔃 Fetching batch of up to ${maxResults} unprocessed images from Firebase Storage`);

    try {
        const options = {
            prefix: 'u_',
            maxResults: maxResults,
            autoPaginate: false
        };

        if (pageToken) {
            options.pageToken = pageToken;
        }

        const [files, , response] = await bucket.getFiles(options);
        const nextPageToken = response?.nextPageToken || null;

        console.log(`✅ Retrieved ${files.length} files from Firebase Storage${nextPageToken ? ' (more available)' : ' (no more files)'}`);

        const unprocessedImages = [];
        let failedImages = [];

        // --- First Pass ---
        for (const file of files) {
            try {
                const [metadata] = await file.getMetadata();
                const customMetadata = metadata.metadata || {};

                const imageData = {
                    id: customMetadata.id,
                    filename: customMetadata.filename,
                    group_id: customMetadata.group_id,
                    created_by_user: customMetadata.user_id,
                    firebase_path: file.name,
                    file_size: metadata.size,
                    content_type: metadata.contentType,
                    uploaded_at: customMetadata.uploaded_at
                };

                if (!imageData.id || !imageData.group_id || !imageData.created_by_user) {
                    console.log(`⚠️ Skipping file ${file.name} - missing required metadata`);
                    failedImages.push(file); // store file for retry
                    continue;
                }

                unprocessedImages.push(imageData);
            } catch (error) {
                console.error(`❌ Error reading metadata for ${file.name}:`, error.message);
                failedImages.push(file);
            }
        }

        // --- Second Retry Pass ---
        if (failedImages.length > 0) {
            console.log(`🔁 Retrying ${failedImages.length} failed images...`);

            const stillFailed = [];
            for (const file of failedImages) {
                try {
                    const [metadata] = await file.getMetadata();
                    const customMetadata = metadata.metadata || {};

                    const imageData = {
                        id: customMetadata.id,
                        filename: customMetadata.filename,
                        group_id: customMetadata.group_id,
                        created_by_user: customMetadata.user_id,
                        firebase_path: file.name,
                        file_size: metadata.size,
                        content_type: metadata.contentType,
                        uploaded_at: customMetadata.uploaded_at
                    };

                    if (!imageData.id || !imageData.group_id || !imageData.created_by_user) {
                        console.log(`⚠️ Skipping file ${file.name} again - still missing metadata`);
                        stillFailed.push(file.name);
                        continue;
                    }

                    unprocessedImages.push(imageData);
                } catch (error) {
                    console.error(`❌ Retry failed for ${file.name}:`, error.message);
                    stillFailed.push(file.name);
                }
            }

            failedImages = stillFailed;
        }

        console.log(`✅ Found ${unprocessedImages.length} valid unprocessed images in this batch`);
        if (failedImages.length > 0) {
            console.warn(`⚠️ ${failedImages.length} images permanently failed after retry:`, failedImages);
        }

        return {
            success: true,
            images: unprocessedImages,
            failedImages,
            filesFetchedFromFirebase: files.length,
            failedImages,
            nextPageToken: nextPageToken,
            hasMore: !!nextPageToken
        };
    } catch (error) {
        console.error('❌ Error fetching images from Firebase:', error.message);
        return {
            success: false,
            error: error.message,
            errorReason: "Not Able to Fetch Images from Firebase",
            images: [],
            nextPageToken: null,
            hasMore: false
        };
    }
}

async function renameFailedFiles(path) {
    try {
        const file = bucket.file("u_" + path);

        const [renamedFile] = await file.rename("f_" + path);

        console.log(`✅ File renamed from ${path} → ${renamedFile.name}`);
        return {
            success: true,
            oldPath: path
        };
    } catch (error) {
        console.error(`❌ Failed to rename file:`, error.message);
        return {
            success: false,
            oldPath: path,
            error: error.message
        };
    }
}

// Process a single image
async function processSingleImage(image, planType) {
    const { id, firebase_path, filename, group_id, created_by_user, uploaded_at } = image;

    console.log(`🔃 Processing Image ${id} for group ${group_id}`);

    try {

        console.log(`🔧 Processing image ${id} from ${firebase_path}`);

        // Download image from Firebase
        const [originalBuffer] = await bucket.file(firebase_path).download();
        console.log(`✅ Downloaded Image ${id} from Firebase`);

        // Get metadata
        const sharpMeta = await sharp(originalBuffer).metadata();
        const originalWidth = sharpMeta.width;
        console.log(`📏 Image ${id} original width: ${originalWidth} px`);
        const originalFormat = sharpMeta.format;

        console.log(`📏 Image ${id} - Format: ${originalFormat}, Width: ${originalWidth}px`);

        // Handle EXIF data (only works for JPEG images)
        let artist = null;
        let dateTaken = null;

        if (originalFormat === 'jpeg' || originalFormat === 'jpg') {
            try {
                const parser = exifParser.create(originalBuffer);
                const result = parser.parse();
                const originalMeta = result.tags;
                artist = originalMeta.Artist || originalMeta.artist || null;
                dateTaken = originalMeta.DateTimeOriginal
                    ? new Date(originalMeta.DateTimeOriginal * 1000)
                    : null;
                console.log(`📏 Image ${id} EXIF - Artist: ${artist}, Date taken: ${dateTaken}`);
            } catch (exifError) {
                console.warn(`⚠️ Could not parse EXIF data for image ${id}: ${exifError.message}`);
                // Continue processing without EXIF data
            }
        } else {
            console.log(`📏 Image ${id} - Non-JPEG format, skipping EXIF extraction`);
        }
        // const parser = exifParser.create(originalBuffer);
        // const result = parser.parse();
        // const originalMeta = result.tags
        // const artist = originalMeta.Artist || originalMeta.artist || null;
        // const dateTaken = originalMeta.DateTimeOriginal
        // ? new Date(originalMeta.DateTimeOriginal * 1000) // exif-parser gives seconds since epoch
        // : null;
        console.log(`📏 Image ${id} original Artist: ${artist}`);
        console.log(`📏 Image ${id} date taken dateTaken: ${dateTaken}`);

        // Strip metadata and rotate
        const baseImage = sharp(originalBuffer);
        const strippedBuffer = await baseImage.rotate().toBuffer();
        let compressedBuffer;
        let compressedBuffer3k;


        console.log(`✅ Stripped metadata for Image ${id}`);

        if (originalWidth <= 1000) {
            // Use stripped buffer as compressed since it's already <= 1000px
            compressedBuffer = strippedBuffer;
            console.log(`✅ Image ${id} is ${originalWidth}px wide(≤ 1000px), using original size`);
        } else {
            // Resize to 1000px
            compressedBuffer = await baseImage.rotate().resize({ width: 1000 }).jpeg().toBuffer();
            console.log(`✅ Resized Image ${id} from ${originalWidth}px to 1000px`);
        }
        const compressedPath = `compressed_${id}`;
        await bucket.file(compressedPath).save(compressedBuffer, {
            contentType: 'image/jpeg',
            metadata: {
                cacheControl: "public, max-age=31536000, immutable"
            },
        });
        const localCompressedPath = path.join(__dirname, "warm-images", `${group_id}`, `compressed_${id}.jpg`);

        // Save locally
        await fs.writeFile(localCompressedPath, compressedBuffer);

        let downloadURLStripped;
        if (planType == 'elite') {
            // Upload stripped image
            const strippedPath = `stripped_${id}`;
            await bucket.file(strippedPath).save(strippedBuffer, {
                contentType: "image/jpeg",
                metadata: {
                    cacheControl: "public, max-age=31536000, immutable"
                },
            });
            console.log(`✅ Stored stripped Image ${id} to Firebase`);
            const strippedFile = bucket.file(strippedPath);

            const [downloadURLStripped_url] = await strippedFile.getSignedUrl({
                action: 'read',
                expires: '03-09-2491'
            });

            downloadURLStripped = downloadURLStripped_url
        }

        let downloadURLCompressed_3k;
        if (planType != 'lite') {
            if (originalWidth <= 3000) {
                // Use stripped buffer as compressed since it's already <= 3000px
                compressedBuffer3k = strippedBuffer;
                console.log(`✅ Image ${id} is ${originalWidth}px wide(≤ 3000px), using original size`);
            } else {
                // Resize to 3000px
                compressedBuffer3k = await baseImage.rotate().resize({ width: 3000 }).jpeg().toBuffer();
                console.log(`✅ Resized Image ${id} from ${originalWidth}px to 3000px`);
            }
            const compressedPath3k = `compressed_3k_${id}`;
            await bucket.file(compressedPath3k).save(compressedBuffer3k, {
                contentType: 'image/jpeg',
                metadata: {
                    cacheControl: "public, max-age=31536000, immutable"
                },
            });

            console.log(`✅ Stored 3000px Image ${id} to Firebase`);

            const compressedFile3k = bucket.file(compressedPath3k);

            const [downloadURLCompressed_3k_url] = await compressedFile3k.getSignedUrl({
                action: 'read',
                expires: '03-09-2491'
            });

            downloadURLCompressed_3k = downloadURLCompressed_3k_url
        }


        // Create 200px thumbnail
        const thumbBuffer = await baseImage.rotate().resize({ width: 200 }).jpeg().toBuffer();
        const thumbPath = `thumbnail_${id}`;
        await bucket.file(thumbPath).save(thumbBuffer, {
            contentType: 'image/jpeg',
            metadata: {
                cacheControl: "public, max-age=31536000, immutable"
            },
        });
        console.log(`✅ Created 200px thumbnail for Image ${id}`);

        const thumbFile = bucket.file(thumbPath);
        const compressedFile = bucket.file(compressedPath);


        // Get signed URLs
        const [downloadURL] = await thumbFile.getSignedUrl({
            action: 'read',
            expires: '03-09-2491' // Far future date for permanent access
        });

        const [downloadURLCompressed] = await compressedFile.getSignedUrl({
            action: 'read',
            expires: '03-09-2491'
        });



        console.log(`✅ Signed URLs generated for image ${id}`);

        return {
            id,
            success: true,
            firebase_path, // Include original path for cleanup
            data: {
                id: id,
                group_id: group_id,
                created_by_user: created_by_user,
                filename: filename,
                uploaded_at: uploaded_at,
                json_meta_data: null,
                thumb_byte: null,
                image_byte: null,
                status: 'warm',
                compressed_location: null,
                artist: artist,
                dateCreated: dateTaken,
                location: downloadURL,
                signedUrl: downloadURLCompressed,
                signedUrl3k: downloadURLCompressed_3k,
                signedUrlStripped: downloadURLStripped
            }
        };
    } catch (error) {
        console.error(`❌ Error processing Image ${id}: `, error.message);
        return {
            id,
            success: false,
            firebase_path, // Include original path even for failed processing
            data: {
                id: id,
                group_id: group_id,
                created_by_user: created_by_user,
                filename: filename,
                uploaded_at: uploaded_at,
                json_meta_data: null,
                thumb_byte: null,
                image_byte: null,
                status: 'warm_failed',
                compressed_location: null,
                artist: null,
                dateCreated: null,
            }
        };
    }
}

// Process images in parallel with concurrency limit
async function processImagesBatch(images, planType, parallel_limit) {
    console.log(`🔃 Processing batch of ${images.length} images with ${parallel_limit} parallel workers`);

    const results = [];

    // Process images in chunks of PARALLEL_LIMIT
    for (let i = 0; i < images.length; i += parallel_limit) {
        const chunk = images.slice(i, i + parallel_limit);
        console.log(`🔃 Processing chunk ${Math.floor(i / parallel_limit) + 1} /${Math.ceil(images.length / parallel_limit)
            } (${chunk.length} images)`);

        const chunkPromises = chunk.map((image) => processSingleImage(image, planType));
        const chunkResults = await Promise.all(chunkPromises);
        results.push(...chunkResults);

        console.log(`✅ Completed chunk ${Math.floor(i / parallel_limit) + 1} /${Math.ceil(images.length / parallel_limit)}`);
    }

    return results;
}

// Delete a single original file with retry logic
async function deleteOriginalFile(firebasePath, retryCount = 0) {
    try {
        const file = bucket.file(firebasePath);

        // Check if file exists before attempting deletion
        const [exists] = await file.exists();
        if (!exists) {
            console.log(`⚠️ File ${firebasePath} no longer exists, skipping deletion`);
            return { success: true, skipped: true };
        }

        await file.delete();
        console.log(`🗑️ Successfully deleted original file: ${firebasePath}`);
        return { success: true, skipped: false };
    } catch (error) {
        console.error(`❌ Error deleting file ${firebasePath} (attempt ${retryCount + 1}): ${error.message}`);

        // Retry logic
        if (retryCount < MAX_CLEANUP_RETRIES) {
            console.log(`🔄 Retrying deletion of ${firebasePath} (attempt ${retryCount + 2}/${MAX_CLEANUP_RETRIES + 1})`);
            await new Promise(resolve => setTimeout(resolve, 1000 * (retryCount + 1))); // Exponential backoff
            return deleteOriginalFile(firebasePath, retryCount + 1);
        }

        return { success: false, error: error.message, skipped: false };
    }
}

// Clean up original files in batches with parallel processing
async function cleanupOriginalFiles(successfulResults) {

    if (successfulResults.length === 0) {
        console.log('⚠️ No successful results to cleanup');
        return { totalAttempted: 0, totalDeleted: 0, totalFailed: 0, totalSkipped: 0 };
    }

    console.log(`🔃 Starting cleanup of ${successfulResults.length} original files`);

    let totalDeleted = 0;
    let totalFailed = 0;
    let totalSkipped = 0;
    let totalAttempted = successfulResults.length;

    // Process cleanup in batches to avoid overwhelming Firebase
    for (let i = 0; i < successfulResults.length; i += CLEANUP_BATCH_SIZE) {
        const batch = successfulResults.slice(i, i + CLEANUP_BATCH_SIZE);
        const batchNumber = Math.floor(i / CLEANUP_BATCH_SIZE) + 1;
        const totalBatches = Math.ceil(successfulResults.length / CLEANUP_BATCH_SIZE);

        console.log(`🔃 Cleaning up batch ${batchNumber}/${totalBatches} (${batch.length} files)`);

        // Create cleanup promises for this batch
        const cleanupPromises = batch.map(result =>
            deleteOriginalFile("u_" + result)
        );

        try {
            const batchResults = await Promise.all(cleanupPromises);

            // Count results for this batch
            const batchDeleted = batchResults.filter(r => r.success && !r.skipped).length;
            const batchSkipped = batchResults.filter(r => r.success && r.skipped).length;
            const batchFailed = batchResults.filter(r => !r.success).length;

            totalDeleted += batchDeleted;
            totalSkipped += batchSkipped;
            totalFailed += batchFailed;

            console.log(`✅ Cleanup batch ${batchNumber}/${totalBatches} completed: ${batchDeleted} deleted, ${batchSkipped} skipped, ${batchFailed} failed`);

            // Log failed deletions for this batch
            const failedDeletions = batch.filter((_, index) => !batchResults[index].success);
            if (failedDeletions.length > 0) {
                console.log(`⚠️ Failed to delete files in batch ${batchNumber}:`,
                    failedDeletions.map(r => `${r.firebase_path} (ID: ${r.id})`));
            }

        } catch (error) {
            console.error(`❌ Error processing cleanup batch ${batchNumber}:`, error.message);
            totalFailed += batch.length;
        }

        // Brief pause between cleanup batches to avoid rate limiting
        if (batchNumber < totalBatches) {
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    }

    console.log(`🧹 Cleanup completed: ${totalDeleted} deleted, ${totalSkipped} skipped, ${totalFailed} failed out of ${totalAttempted} attempted`);

    // Log summary of failed deletions if any
    if (totalFailed > 0) {
        console.warn(`⚠️ ${totalFailed} files could not be deleted after ${MAX_CLEANUP_RETRIES + 1} attempts. These may need manual cleanup.`);
    }

    return {
        totalAttempted,
        totalDeleted,
        totalFailed,
        totalSkipped,
        cleanupSuccess: totalFailed === 0
    };
}

// Verify database insertion was successful by checking if records exist
async function verifyDatabaseInsertion(client, results) {
    const successfulResults = results.filter(r => r.success);

    if (successfulResults.length === 0) {
        return { verified: true, missingIds: [] };
    }

    console.log(`🔍 Verifying database insertion for ${successfulResults.length} records`);

    try {
        const ids = successfulResults.map(r => r.id);
        const placeholders = ids.map((_, index) => `$${index + 1}`).join(', ');

        const { rows } = await client.query(
            `SELECT id FROM images WHERE id IN (${placeholders})`,
            ids
        );

        const existingIds = new Set(rows.map(row => row.id));
        const missingIds = ids.filter(id => !existingIds.has(id));

        if (missingIds.length === 0) {
            console.log(`✅ Database verification successful: All ${successfulResults.length} records found`);
            return { verified: true, missingIds: [] };
        } else {
            console.warn(`⚠️ Database verification failed: ${missingIds.length} records missing from database`);
            console.log(`Missing IDs: ${missingIds.join(', ')}`);
            return { verified: false, missingIds };
        }
    } catch (error) {
        console.error(`❌ Error verifying database insertion:`, error.message);
        return { verified: false, missingIds: [], error: error.message };
    }
}

// Attempt single batch insert
async function performBatchInsert(client, successfulResults) {
    try {
        const valuesClauses = [];
        const allParams = [];
        let paramIndex = 1;

        for (const result of successfulResults) {
            const { data } = result;

            valuesClauses.push(
                `($${paramIndex}::uuid, $${paramIndex + 1}, $${paramIndex + 2}, $${paramIndex + 3}, $${paramIndex + 4}::timestamp, $${paramIndex + 5}, $${paramIndex + 6}::jsonb, $${paramIndex + 7}::bytea, $${paramIndex + 8}::bytea, $${paramIndex + 9}, $${paramIndex + 10}, $${paramIndex + 11}::timestamp, $${paramIndex + 12}, $${paramIndex + 13}, $${paramIndex + 14} , $${paramIndex + 15} , NOW())`
            );

            allParams.push(
                data.id,
                data.group_id,
                data.created_by_user,
                data.filename,
                data.uploaded_at,
                data.status,
                data.json_meta_data,
                data.thumb_byte,
                data.image_byte,
                data.compressed_location,
                data.artist,
                data.dateCreated,
                data.location,
                data.signedUrl,
                data.signedUrl3k,
                data.signedUrlStripped
            );

            paramIndex += 16;
        }

        const batchInsertQuery = `
        INSERT INTO images 
        (id, group_id, created_by_user, filename, uploaded_at, status, json_meta_data, thumb_byte, image_byte, compressed_location, artist, date_taken, location, signed_url, signed_url_3k , signed_url_stripped ,last_processed_at)
        VALUES ${valuesClauses.join(', ')}
        ON CONFLICT (id) DO UPDATE SET
            status = EXCLUDED.status,
            json_meta_data = EXCLUDED.json_meta_data,
            thumb_byte = EXCLUDED.thumb_byte,
            image_byte = EXCLUDED.image_byte,
            compressed_location = EXCLUDED.compressed_location,
            artist = EXCLUDED.artist,
            date_taken = EXCLUDED.date_taken,
            last_processed_at = NOW(),
            location = EXCLUDED.location,
            signed_url = EXCLUDED.signed_url,
            signed_url_3k = EXCLUDED.signed_url_3k,
            signed_url_stripped = EXCLUDED.signed_url_stripped
    `;

        console.log(`🔃 Executing batch insert query with ${allParams.length} parameters...`);
        const startTime = Date.now();

        const timeoutPromise = new Promise((_, reject) => {
            setTimeout(() => reject(new Error('Batch insert timeout after 180 seconds')), 180000);
        });

        const insertResult = await Promise.race([
            client.query(batchInsertQuery, allParams),
            timeoutPromise
        ]);

        const duration = Date.now() - startTime;
        console.log(`✅ Successfully batch inserted ${insertResult.rowCount} images in database (${duration}ms)`);

        if (insertResult.rowCount !== successfulResults.length) {
            console.warn(`⚠️ Expected to insert ${successfulResults.length} records, but inserted ${insertResult.rowCount}`);
        }

        return {
            success: true,
            insertedIds: successfulResults.map(r => r.data.id),
            failedIds: [],
            rowCount: insertResult.rowCount
        };
    } catch (error) {
        return {
            success: false,
            errorReason: "Batch Update Faild : " + error.message,
            insertedIds: [],
            failedIds: successfulResults.map(r => r.data.id)
        };
    }

}

// Fallback: chunked inserts (smaller batches)
async function performChunkedInserts(client, successfulResults) {
    const CHUNK_SIZE = 5;
    console.log(`🔃 Using chunked inserts with ${CHUNK_SIZE} records per chunk`);

    let totalInserted = 0;
    const insertedIds = [];
    const failedIds = []
    await client.query('BEGIN');
    try {
        for (let i = 0; i < successfulResults.length; i += CHUNK_SIZE) {
            const chunk = successfulResults.slice(i, i + CHUNK_SIZE);
            console.log(`🔃 Inserting chunk ${Math.floor(i / CHUNK_SIZE) + 1} /${Math.ceil(successfulResults.length / CHUNK_SIZE)}`);

            for (const result of chunk) {
                const { data } = result;
                try {
                    const insertResult = await client.query(
                        `INSERT INTO images
                         (id, group_id, created_by_user, filename, uploaded_at, status, json_meta_data, thumb_byte, image_byte, compressed_location, artist, date_taken, location, signed_url, signed_url_3k, signed_url_stripped , last_processed_at)
                         VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15 , $16 ,NOW())
                         ON CONFLICT (id) DO UPDATE SET
                            status = EXCLUDED.status,
                            json_meta_data = EXCLUDED.json_meta_data,
                            thumb_byte = EXCLUDED.thumb_byte,
                            image_byte = EXCLUDED.image_byte,
                            compressed_location = EXCLUDED.compressed_location,
                            artist = EXCLUDED.artist,
                            date_taken = EXCLUDED.date_taken,
                            last_processed_at = NOW(),
                            location = EXCLUDED.location,
                            signed_url = EXCLUDED.signed_url,
                            signed_url_3k = EXCLUDED.signed_url_3k,
                            signed_url_stripped = EXCLUDED.signed_url_stripped`,
                        [data.id, data.group_id, data.created_by_user, data.filename, data.uploaded_at, data.status, data.json_meta_data, data.thumb_byte, data.image_byte, data.compressed_location, data.artist, data.dateCreated, data.location, data.signedUrl, data.signedUrl3k, data.signedUrlStripped]
                    );
                    if (insertResult.rowCount > 0) {
                        insertedIds.push(data.id);
                        totalInserted += insertResult.rowCount;
                    } else {
                        failedIds.push(data.id);
                    }
                } catch (error) {
                    console.error(`❌ Failed to insert image ${data.id} in chunk: `, error.message);
                    failedIds.push(data.id);
                }
            }

            console.log(`✅ Inserted chunk ${Math.floor(i / CHUNK_SIZE) + 1} /${Math.ceil(successfulResults.length / CHUNK_SIZE)} (${totalInserted}/${successfulResults.length} total)`);
        }
        await client.query('COMMIT');
    } catch (error) {
        await client.query('ROLLBACK');
        console.log("❌ Chunk insert error:", error.message);
        return { success: false, errorReason: "Chuck Update Failed : " + error.message, insertedIds: [] }
    }

    console.log(`✅ Successfully inserted ${insertedIds.length} images using chunked approach`);
    if (failedIds.length) {
        console.warn(`⚠️ Failed to insert ${failedIds.length} images:`, failedIds);
    }

    return { success: true, insertedIds, failedIds };
}

// Insert into database with all results at once using batch INSERT with fallback
async function insertIntoDatabaseBatch(client, results, run_id) {
    const successfulResults = results.filter(r => r.success);
    const failedResults = results.filter(r => !r.success);

    if (failedResults.length > 0) {
        console.log(`⚠️  ${failedResults.length} images failed processing: `,
            failedResults.map(r => `ID: ${r.id}, Error: ${r.error} `));
    }

    if (successfulResults.length === 0) {
        console.log('⚠️  No successful results to insert in database');
        return { success: true, failedIds: [], insertedIds: [] }
    }

    console.log(`🔃 Batch inserting database for ${successfulResults.length} successfully processed images`);


    // Try batch insert first
    try {
        console.log(`🔃 Attempting single batch INSERT query...`);
        await client.query('BEGIN');
        // const batchInsertResponse = await performChunkedInserts(client, successfulResults);
        const batchInsertResponse = await performBatchInsert(client, successfulResults);
        await client.query('COMMIT');

        if (!batchInsertResponse.success) {
            console.log(`⚠️  Batch INSERT failed: ${batchInsertResponse.errorReason} `);
            await updateStatusHistory(client, run_id, "node_compression", "batch_db_insert", successfulResults.length, batchInsertResponse.failedIds.length, batchInsertResponse.insertedIds.length, successfulResults[0].group_id,
                batchInsertResponse.errorReason + "\n trying chuck update for ids: " + batchInsertResponse.failedIds.join(", \n"))
            // try {
            //     console.log(`🔃 Starting Rollback`);
            //     const rollbackPromise = client.query('ROLLBACK');
            //     const rollbackTimeout = new Promise((_, reject) => {
            //         setTimeout(() => reject(new Error('Rollback timeout after 180 seconds')), 0);
            //     });
            //     await Promise.race([rollbackPromise, rollbackTimeout]);
            //     console.log(`✅ Rollback completed`);
            // } catch (rollbackError) {
            //     console.log(`⚠️  Rollback error: ${rollbackError.message} `);
            //     new ProcessingError("Not able to do db roll back", {
            //         reason: "Not able to do db roll back " + rollbackError.message
            //     })
            // }

            // Try chunked inserts
            try {
                console.log(`🔃 Falling back to chunked inserts...`);
                const chunkInsertResponse = await performChunkedInserts(client, successfulResults);
                if (!chunkInsertResponse.success) {
                    await updateStatusHistory(client, run_id, "node_compression", "chunk_db_insert", successfulResults.length, chunkInsertResponse.failedIds.length, chunkInsertResponse.insertedIds.length, successfulResults[0].group_id,
                        chunkInsertResponse.errorReason + "ids: " + chunkInsertResponse.failedIds.join(", \n"))
                    return { success: false, failedIds: successfulResults.map(r => r.id) }
                }
                return { success: true, failedIds: chunkInsertResponse.failedIds, insertedIds: chunkInsertResponse.insertedIds }

            } catch (chunkError) {
                console.log(`⚠️  Chunk INSERT failed: ${chunkError.message} `);

                await updateStatusHistory(client, run_id, "node_compression", "chunk_db_insert", successfulResults.length, null, null, successfulResults[0].group_id,
                    chunkError.message + "ids: " + successfulResults.map(r => r.id).join(", \n"))
                return { success: false, failedIds: successfulResults.map(r => r.id), error: chunkError.message, insertedIds: [] }

            }
        }
        return { success: true, failedIds: batchInsertResponse.failedIds, insertedIds: batchInsertResponse.insertedIds }
    } catch (batchError) {
        await updateStatusHistory(client, run_id, "node_compression", "batch_db_insert", successfulResults.length, null, null, successfulResults[0].group_id,
            batchError.message + "db update failed for ids: " + successfulResults.map(r => r.id).join(", \n"))
        return { success: false, failedIds: successfulResults.map(r => r.id), error: batchError.message }
    }
}

// Process images in batches of BATCH_SIZE with database insert after each batch
async function processImagesBatches(client, images, planType, groupId, run_id) {
    console.log(`🔃 Processing ${images.length} images in batches of ${BATCH_SIZE}`);
    let totalCleaned = 0;
    let totalCleanupFailed = 0;
    let allResults = [];
    let totalImagesInsertedIntoDB = []
    let totalImagesFailedDBInsertion = []
    // Process images in batches of BATCH_SIZE
    for (let i = 0; i < images.length; i += BATCH_SIZE) {
        const batch = images.slice(i, i + BATCH_SIZE);
        const batchNumber = Math.floor(i / BATCH_SIZE) + 1;
        const totalBatches = Math.ceil(images.length / BATCH_SIZE);

        console.log(`🔃 Processing batch ${batchNumber}/${totalBatches} (${batch.length} images)`);
        // const parallelLimit = await getDynamicParallelLimit(client);
        const parallelLimit = 10;

        // Process this batch
        const batchResults = await processImagesBatch(batch, planType, parallelLimit);
        allResults.push(...batchResults);

        // Insert into database immediately after processing this batch
        console.log(`🔃 Inserting database for batch ${batchNumber}/${totalBatches}`);
        const insertionResult = await insertIntoDatabaseBatch(client, batchResults, run_id);

        totalImagesInsertedIntoDB.push(...insertionResult.insertedIds);
        totalImagesFailedDBInsertion.push(...insertionResult.failedIds);
        totalImagesFailedDBInsertion.push(...batchResults.filter(r => !r.success));

        for (const id of insertionResult.failedIds) {
            try {
                const res = await renameFailedFiles(id)
                if (!res.success) {
                    await updateStatusHistory(client, run_id, "node_compression", "failed_renaming", insertionResult.failedIds.length, null, null, groupId, "Failed Files cannot be renamed")
                    throw new ProcessingError("Cannot rename failed files", {
                        groupId: groupId,
                        reason: "Cannot rename failed files : " + res.error
                    })
                }

            } catch (error) {
                await updateStatusHistory(client, run_id, "node_compression", "failed_renaming", insertionResult.failedIds.length, null, null, groupId, "Failed Files cannot be renamed")
                throw new ProcessingError("Cannot rename failed files", {
                    groupId: groupId,
                    reason: "Cannot rename failed files : " + error.message
                })
            }
        }


        for (const record of batchResults.filter(r => !r.success)) {
            try {
                const res = await renameFailedFiles(record.id)
                if (!res.success) {
                    await updateStatusHistory(client, run_id, "node_compression", "failed_renaming", insertionResult.failedIds.length, null, null, groupId, "Failed Files cannot be renamed")
                    throw new ProcessingError("Cannot rename failed files", {
                        groupId: groupId,
                        reason: "Cannot rename failed files : " + res.error
                    })
                }
            } catch (error) {
                await updateStatusHistory(client, run_id, "node_compression", "failed_renaming", insertionResult.failedIds.length, null, null, groupId, "Failed Files cannot be renamed")
                throw new ProcessingError("Cannot rename failed files", {
                    groupId: groupId,
                    reason: "Cannot rename failed files : " + error.message
                })
            }
        }
        // Only cleanup if database insertion was successful
        if (insertionResult.success) {
            console.log(`🔃 Cleaning up original files for batch ${batchNumber}/${totalBatches}`);

            const cleanupResult = await cleanupOriginalFiles(insertionResult.insertedIds);

            totalCleaned += cleanupResult.totalDeleted;
            totalCleanupFailed += cleanupResult.totalFailed;

            if (!cleanupResult.cleanupSuccess) {
                console.warn(`⚠️ Some files in batch ${batchNumber}/${totalBatches} could not be cleaned up`);
                await updateStatusHistory(client, run_id, "node_compression", "cleaning", insertionResult.insertedIds.length, null, null, groupId, "Some files in batch could not be cleaned up")
            }
        } else {
            console.log(`⚠️ Skipping cleanup for batch ${batchNumber}/${totalBatches} due to database insertion failure`);
            throw new ProcessingError("Failed to do database insert", {
                reason: "Failed to do database insert : " + insertionResult.error
            })
        }


        console.log(`✅ Completed batch ${batchNumber}/${totalBatches}`);

        // Optional: Add a small delay between batches
        if (batchNumber < totalBatches) {
            console.log(`⏸️  Brief pause before next batch...`);
            await new Promise(resolve => setTimeout(resolve, 100));
        }
    }

    console.log(`🎉 All batches completed. Final stats`);

    return {
        totalCleaned,
        totalCleanupFailed,
        allResults,
        totalImagesInsertedIntoDB,
        totalImagesFailedDBInsertion
    };
}

// Get plan type for a group
async function getGroupPlanType(client, groupIds) {
    try {

        const { rows } = await client.query(
            `SELECT id, plan_type 
     FROM groups
     WHERE id = ANY($1)`,
            [groupIds]
        );

        // Map to [{ groupId, planType }]
        planDeatils = rows.map(r => ({
            groupId: r.id,
            planType: r.plan_type
        }));
        return {
            success: true,
            planDeatils: planDeatils
        }
    } catch (error) {
        console.error("Error getting play type:", error);

        // optional: rethrow to let caller handle it
        // throw error;

        // or return a failure object
        return { success: false, errorReason: "getting plan type for groups", error: error.message };
    }
}

async function updateStatusHistory(client, run_id, task, sub_task, totalImagesInitialized, totalImagesFailed, totalImagesProcessed, groupId, fail_reason) {
    try {
        await client.query(
            `INSERT INTO process_history 
        (worker_id,run_id , task,sub_task, initialized_count, success_count, failed_count, group_id, ended_at , fail_reason)
       VALUES 
        (1 ,$7, $6,$8, $1, $2, $3, $4, NOW() , $5)`,
            [totalImagesInitialized, totalImagesProcessed, totalImagesFailed, groupId, fail_reason, task, run_id, sub_task]
        );

        return { success: true };
    } catch (error) {
        console.error("Error inserting into process_history:", error);

        // optional: rethrow to let caller handle it
        // throw error;

        // or return a failure object
        return { success: false, errorReason: "updating history", error: error.message };
    }
}

async function updateStatus(client, groupId, failReason, isIdeal) {
    try {
        await client.query(
            `UPDATE process_status 
             SET task_status = 'failed', 
                 processing_group = $1, 
                 fail_reason = $2, 
                 ended_at = NOW(), 
                 is_ideal = $3
             WHERE task = 'node_compression'`,
            [groupId, failReason, isIdeal]
        );
        return { success: true };
    } catch (error) {
        console.error("Error updating process status:", error);
        return { success: false, errorReason: "updating status", error: error.message };
    }
}


// Enhanced cleanup function for emergency/manual cleanup
async function emergencyCleanupOriginalFiles(client, dryRun = true) {
    console.log(`🚨 Starting ${dryRun ? 'DRY RUN' : 'EMERGENCY'} cleanup of orphaned original files`);

    try {
        // Get all successfully processed images from database
        const { rows: processedImages } = await client.query(
            `SELECT id FROM images WHERE status IN ('warm', 'cold') AND id IS NOT NULL`
        );

        const processedIds = new Set(processedImages.map(row => row.id));
        console.log(`📊 Found ${processedIds.size} successfully processed images in database`);

        // Get all original files from Firebase
        const [originalFiles] = await bucket.getFiles({ prefix: 'u_' });
        console.log(`📊 Found ${originalFiles.length} original files in Firebase Storage`);

        const orphanedFiles = [];

        for (const file of originalFiles) {
            try {
                const [metadata] = await file.getMetadata();
                const customMetadata = metadata.metadata || {};
                const imageId = customMetadata.id;

                if (imageId && processedIds.has(imageId)) {
                    // This file has been successfully processed
                    orphanedFiles.push({
                        path: file.name,
                        id: imageId,
                        size: metadata.size
                    });
                }
            } catch (error) {
                console.error(`❌ Error checking file ${file.name}:`, error.message);
            }
        }

        console.log(`📊 Found ${orphanedFiles.length} orphaned original files that can be safely deleted`);

        if (dryRun) {
            console.log(`🔍 DRY RUN: Would delete the following files:`);
            orphanedFiles.forEach(file => {
                console.log(`  - ${file.path} (ID: ${file.id}, Size: ${file.size} bytes)`);
            });

            const totalSize = orphanedFiles.reduce((sum, file) => sum + parseInt(file.size || 0), 0);
            console.log(`📊 Total space that would be freed: ${(totalSize / 1024 / 1024).toFixed(2)} MB`);

            return {
                totalFound: orphanedFiles.length,
                totalSize,
                dryRun: true
            };
        } else {
            console.log(`🗑️ Proceeding with deletion of ${orphanedFiles.length} orphaned files...`);

            // Create mock results for cleanup function
            const mockResults = orphanedFiles.map(file => ({
                success: true,
                firebase_path: file.path,
                id: file.id
            }));

            const cleanupResult = await cleanupOriginalFiles(mockResults);
            return {
                totalFound: orphanedFiles.length,
                cleanupResult,
                dryRun: false
            };
        }
    } catch (error) {
        console.error(`❌ Error during emergency cleanup:`, error.message);
        throw error;
    }
}

async function processImages() {
    const client = await pool.connect();

    try {
        while (true) {
            const run_id = Date.now()
            await updateStatusHistory(client, run_id, "node_compression", "run", null, null, null, null, "")
            console.log("🔃 Starting new processing cycle");

            let pageToken = null;
            let hasMoreImages = true;
            let totalProcessedAllBatches = 0;
            let totalCleanedAllBatches = 0;
            let totalBatchesProcessed = 0;
            let totalAllImagesProcessed = [];
            let totalAllImagesProcessFailed = [];
            let totalImagesInitialized = 0;
            let totalMetaDataFailedImages = [];
            // Keep fetching and processing batches until no more images
            while (hasMoreImages) {
                console.log(`🔃 Fetching batch ${totalBatchesProcessed + 1} from Firebase Storage`);

                // Fetch batch of unprocessed images from Firebase with proper pagination
                const resFromFirebaseFetch = await fetchUnprocessedImagesBatch(FIREBASE_BATCH_SIZE, pageToken);
                totalImagesInitialized += resFromFirebaseFetch.filesFetchedFromFirebase
                totalMetaDataFailedImages.push(...resFromFirebaseFetch.failedImages)
                if (!resFromFirebaseFetch.success) {
                    throw new ProcessingError(resFromFirebaseFetch.error, {
                        reason: resFromFirebaseFetch.errorReason + " : " + resFromFirebaseFetch.error,
                        retryable: true,
                    });
                }

                const { images: unprocessedImages, nextPageToken, hasMore } = resFromFirebaseFetch;
                if (unprocessedImages.length === 0) {
                    console.log("⏸️ No more unprocessed images found in this batch");
                    hasMoreImages = false;
                    break;
                }

                // Group images by group_id
                const imagesByGroup = {};
                for (const image of unprocessedImages) {
                    if (!imagesByGroup[image.group_id]) {
                        imagesByGroup[image.group_id] = [];
                    }
                    imagesByGroup[image.group_id].push(image);
                }

                console.log(`✅ Batch ${totalBatchesProcessed + 1}: Found ${unprocessedImages.length} images in ${Object.keys(imagesByGroup).length} groups`);

                let batchProcessedCount = 0;
                let batchCleanedCount = 0;
                const planTypesResonse = await getGroupPlanType(client, Object.keys(imagesByGroup));
                if (!planTypesResonse.success) {
                    throw new ProcessingError(planTypesResonse.error, {
                        groupId: null,
                        reason: planTypesResonse.errorReason + " : " + planTypesResonse.error,
                        retryable: false,
                    });
                }
                // Process each group in this batch
                for (const [groupId, groupImages] of Object.entries(imagesByGroup)) {
                    await fs.mkdir(path.join(__dirname, "warm-images", `${groupId}`), { recursive: true });
                    console.log(`🔃 Processing ${groupImages.length} images for group ${groupId} (Batch ${totalBatchesProcessed + 1})`);

                    // Get plan type for this group
                    const planTypeEntry = planTypesResonse.planDeatils.find(p => p.groupId == groupId);
                    // const planType = planTypeEntry.planType;
                    const planType = planTypeEntry.planType;
                    console.log(`📋 Group ${groupId} plan type: ${planType}`);
                    if (!["lite", "elite", "pro"].includes(planType)) {
                        updateStatusHistory(client, run_id, "node_compression", "group", totalImagesInitialized, totalImagesInitialized - totalAllImagesProcessed.length, totalAllImagesProcessed.length, groupId, "Plan Type not found for group " + groupId)
                        continue;
                    }


                    // Process all images in this group
                    const { totalProcessed, totalCleaned, allResults, totalImagesInsertedIntoDB, totalImagesFailedDBInsertion } = await processImagesBatches(client, groupImages, planType, groupId, run_id);
                    totalAllImagesProcessed.push(...totalImagesInsertedIntoDB.map(r => r.id));
                    totalAllImagesProcessFailed.push(...totalImagesFailedDBInsertion.map(r => r.id));
                    await updateStatusHistory(client, run_id, "node_compression", "group", groupImages.length, totalImagesFailedDBInsertion.length, totalImagesInsertedIntoDB.length, groupId, "DB insertion failed for " + totalImagesFailedDBInsertion.join(" , \n") + " \n")
                    batchProcessedCount += totalProcessed;
                    batchCleanedCount += totalCleaned;

                    console.log(`✅ Completed processing for group ${groupId}. Processed: ${totalProcessed}/${groupImages.length}, Cleaned: ${totalCleaned}`);
                }

                totalProcessedAllBatches += batchProcessedCount;
                totalCleanedAllBatches += batchCleanedCount;
                totalBatchesProcessed++;

                console.log(`🎉 Finished processing batch ${totalBatchesProcessed}. Batch processed: ${batchProcessedCount}, Batch cleaned: ${batchCleanedCount}, Total processed: ${totalProcessedAllBatches}, Total cleaned: ${totalCleanedAllBatches}`);

                // Update pagination state
                pageToken = nextPageToken;
                hasMoreImages = hasMore;

                // Brief pause between Firebase batches to avoid overwhelming the system
                if (hasMoreImages) {
                    console.log(`⏸️ Brief pause before fetching next batch...`);
                    await new Promise(resolve => setTimeout(resolve, 1000)); // 1 second pause
                }
            }

            if (totalProcessedAllBatches === 0) {
                console.log("⏸️ No unprocessed images found in entire cycle, waiting...");
                await updateStatusHistory(client, run_id, "node_compression", "run", totalImagesInitialized, totalImagesInitialized - totalAllImagesProcessed.length, totalAllImagesProcessed.length, null, "DB insertion failed for " + totalAllImagesProcessFailed.join(" , \n") + " \n" + "Meta Data Not Fetched:" + totalMetaDataFailedImages.join(" , \n"))
                await new Promise(res => setTimeout(res, 300000)); // wait 5 minutes
            } else {
                console.log(`🎉 Completed full processing cycle. Total batches: ${totalBatchesProcessed}, Total processed: ${totalProcessedAllBatches}, Total files cleaned up: ${totalCleanedAllBatches}`);

                // Short pause before starting next cycle
                console.log(`⏸️ Brief pause before starting next cycle...`);
                await new Promise(resolve => setTimeout(resolve, 5000)); // 5 second pause
            }
        }
    } catch (err) {
        if (err instanceof ProcessingError) {
            console.error(`❌ ProcessingError: ${err.message}`, {
                groupId: err.groupId,
                reason: err.reason,
            });

            // call your DB status update in one place
            await updateStatus(client, null, err.reason, true);

        } else {
            // unexpected error
            console.error("❌ Unexpected error:", err);
            await updateStatus(client, null, err.message, true);
            console.log("⏸️ Waiting before retry...");
            await new Promise(resolve => setTimeout(resolve, 30000));
        }
    } finally {
        client.release();
        await pool.end();
    }
}

// Function to run emergency cleanup (can be called separately)
async function runEmergencyCleanup(dryRun = true) {
    const client = await pool.connect();
    try {
        const result = await emergencyCleanupOriginalFiles(client, dryRun);
        console.log('🎉 Emergency cleanup completed:', result);
        return result;
    } catch (error) {
        console.error('❌ Emergency cleanup failed:', error);
        throw error;
    } finally {
        client.release();
        await pool.end();
    }
}

// Run the main process if this file is executed directly
if (require.main === module) {
    processImages();
}