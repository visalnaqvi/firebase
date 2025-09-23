const { Pool } = require('pg');
const admin = require('firebase-admin');
const sharp = require('sharp');
const path = require('path');
const serviceAccount = require('./firebase-key.json');
const exifParser = require('exif-parser');
const { error } = require('console');
const fs = require("fs").promises;
const { google } = require('googleapis'); // Add Google APIs client
const axios = require('axios'); // For HTTP requests

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
const FIREBASE_BATCH_SIZE = 50;
const CLEANUP_BATCH_SIZE = 20;
const MAX_CLEANUP_RETRIES = 3;

// Google Drive API helper functions
async function refreshAccessToken(refreshToken, clientId, clientSecret) {
    try {
        const response = await axios.post('https://oauth2.googleapis.com/token', {
            refresh_token: refreshToken,
            client_id: clientId,
            client_secret: clientSecret,
            grant_type: 'refresh_token'
        });

        return {
            success: true,
            accessToken: response.data.access_token,
            expiresIn: response.data.expires_in
        };
    } catch (error) {
        console.error('Error refreshing access token:', error.response?.data || error.message);
        return {
            success: false,
            error: error.response?.data?.error || error.message
        };
    }
}

async function getValidAccessToken(client, userId, groupId) {
    try {
        // Get user tokens from database
        const { rows } = await client.query(
            'SELECT access_token, refresh_token, token_expires_at FROM users WHERE id = $1',
            [userId]
        );

        if (rows.length === 0) {
            throw new Error(`User ${userId} not found`);
        }

        const { access_token, refresh_token, token_expires_at } = rows[0];

        // Check if access token is still valid (with 5 minute buffer)
        const now = new Date();
        const expiresAt = new Date(token_expires_at);
        const bufferTime = 5 * 60 * 1000; // 5 minutes in milliseconds

        if (expiresAt - now > bufferTime) {
            return { success: true, accessToken: access_token };
        }

        // Need to refresh the token
        console.log(`🔄 Refreshing access token for user ${userId}`);
        const refreshResult = await refreshAccessToken(
            refresh_token,
            id, // You'll need to add these to your environment
            s
        );

        if (!refreshResult.success) {
            throw new Error(`Failed to refresh token: ${refreshResult.error}`);
        }

        // Update the database with new access token
        const newExpiresAt = new Date(Date.now() + (refreshResult.expiresIn * 1000));
        await client.query(
            'UPDATE users SET access_token = $1, token_expires_at = $2 WHERE id = $3',
            [refreshResult.accessToken, newExpiresAt, userId]
        );

        console.log(`✅ Access token refreshed for user ${userId}`);
        return { success: true, accessToken: refreshResult.accessToken };

    } catch (error) {
        console.error(`❌ Error getting valid access token for user ${userId}:`, error.message);
        return { success: false, error: error.message };
    }
}

async function listDriveFiles(accessToken, folderId) {
    try {
        const drive = google.drive({ version: 'v3' });

        // Set up OAuth2 client with the access token
        const auth = new google.auth.OAuth2();
        auth.setCredentials({ access_token: accessToken });

        const response = await drive.files.list({
            auth: auth,
            q: `'${folderId}' in parents and mimeType contains 'image/' and trashed=false`,
            fields: 'files(id,name,size,mimeType,createdTime,modifiedTime)',
            pageSize: 1000 // Adjust as needed
        });

        return {
            success: true,
            files: response.data.files
        };

    } catch (error) {
        console.error('Error listing Drive files:', error);
        return {
            success: false,
            error: error.message
        };
    }
}

async function downloadFileFromDrive(accessToken, fileId) {
    try {
        const drive = google.drive({ version: 'v3' });

        const auth = new google.auth.OAuth2();
        auth.setCredentials({ access_token: accessToken });

        const response = await drive.files.get({
            auth: auth,
            fileId: fileId,
            alt: 'media'
        }, {
            responseType: 'stream'
        });

        // Convert stream to buffer
        const chunks = [];
        for await (const chunk of response.data) {
            chunks.push(chunk);
        }

        return {
            success: true,
            buffer: Buffer.concat(chunks)
        };

    } catch (error) {
        console.error('Error downloading file from Drive:', error);
        return {
            success: false,
            error: error.message
        };
    }
}

// Modified function to fetch unprocessed folders from database instead of Firebase
async function fetchUnprocessedFoldersBatch(client, maxResults = FIREBASE_BATCH_SIZE, offset = 0) {
    console.log(`🔃 Fetching batch of up to ${maxResults} unprocessed folders from drive_folders table`);

    try {
        const { rows: folders } = await client.query(
            `SELECT df.folder_id , df.group_id, df.user_id, df.id as drive_folder_id
             FROM drive_folders df 
             WHERE df.is_processed = false 
             ORDER BY df.created_at 
             LIMIT $1 OFFSET $2`,
            [maxResults, offset]
        );

        if (folders.length === 0) {
            return {
                success: true,
                folders: [],
                hasMore: false
            };
        }

        console.log(`✅ Retrieved ${folders.length} unprocessed folders`);

        const processedFolders = [];

        for (const folder of folders) {
            try {
                // Get valid access token for this user
                const tokenResult = await getValidAccessToken(client, folder.user_id, folder.group_id);
                if (!tokenResult.success) {
                    console.error(`❌ Cannot get access token for user ${folder.user_id}: ${tokenResult.error}`);
                    continue;
                }
                console.log("got a token")
                // List files in the Drive folder
                const filesResult = await listDriveFiles(tokenResult.accessToken, folder.folder_id);
                if (!filesResult.success) {
                    console.error(`❌ Cannot list files in folder ${folder.folder_id}: ${filesResult.error}`);
                    continue;
                }
                console.log("got file", filesResult.length)
                // Transform Drive files to match expected format
                const images = filesResult.files.map(file => ({
                    id: file.id, // Use Drive file ID
                    filename: file.name,
                    group_id: folder.group_id,
                    created_by_user: folder.user_id,
                    file_size: parseInt(file.size) || 0,
                    content_type: file.mimeType,
                    uploaded_at: file.createdTime,
                    folder_id: folder.folder_id,
                    drive_folder_id: folder.drive_folder_id,
                    access_token: tokenResult.accessToken
                }));
                console.log("got images", images.length)
                processedFolders.push({
                    ...folder,
                    images: images,
                    access_token: tokenResult.accessToken
                });

            } catch (error) {
                console.error(`❌ Error processing folder ${folder.folder_id}:`, error.message);
            }
        }

        // Check if there are more folders
        const { rows: countRows } = await client.query(
            'SELECT COUNT(*) as total FROM drive_folders WHERE is_processed = false'
        );
        const totalUnprocessed = parseInt(countRows[0].total);
        const hasMore = (offset + maxResults) < totalUnprocessed;

        return {
            success: true,
            folders: processedFolders,
            hasMore: hasMore
        };

    } catch (error) {
        console.error('❌ Error fetching folders from database:', error.message);
        return {
            success: false,
            error: error.message,
            errorReason: "Not Able to Fetch Folders from Database",
            folders: []
        };
    }
}

// Modified function to process single image from Google Drive
async function processSingleImageFromDrive(image, planType) {
    const { id, drive_path, filename, group_id, created_by_user, uploaded_at, access_token } = image;

    console.log(`🔃 Processing Image ${id} from Google Drive for group ${group_id}`);

    try {
        console.log(`🔧 Processing image ${id} from Drive file ${drive_path}`);

        // Download image from Google Drive instead of Firebase
        const downloadResult = await downloadFileFromDrive(access_token, drive_path);
        if (!downloadResult.success) {
            throw new Error(`Failed to download from Drive: ${downloadResult.error}`);
        }

        const originalBuffer = downloadResult.buffer;
        console.log(`✅ Downloaded Image ${id} from Google Drive`);

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
            }
        } else {
            console.log(`📏 Image ${id} - Non-JPEG format, skipping EXIF extraction`);
        }

        console.log(`📏 Image ${id} original Artist: ${artist}`);
        console.log(`📏 Image ${id} date taken dateTaken: ${dateTaken}`);

        // Strip metadata and rotate
        const baseImage = sharp(originalBuffer);
        const strippedBuffer = await baseImage.rotate().toBuffer();
        let compressedBuffer;
        let compressedBuffer3k;

        console.log(`✅ Stripped metadata for Image ${id}`);

        if (originalWidth <= 1000) {
            compressedBuffer = strippedBuffer;
            console.log(`✅ Image ${id} is ${originalWidth}px wide(≤ 1000px), using original size`);
        } else {
            compressedBuffer = await baseImage.rotate().resize({ width: 1000 }).jpeg().toBuffer();
            console.log(`✅ Resized Image ${id} from ${originalWidth}px to 1000px`);
        }

        // Upload compressed image to Firebase
        const compressedPath = `compressed_${id}`;
        await bucket.file(compressedPath).save(compressedBuffer, {
            contentType: 'image/jpeg',
            metadata: {
                cacheControl: "public, max-age=31536000, immutable",
                metadata: {
                    id: id,
                    filename: filename,
                    group_id: group_id,
                    user_id: created_by_user,
                    uploaded_at: uploaded_at,
                    source: 'google_drive',
                    drive_file_id: drive_path
                }
            },
        });

        const localCompressedPath = path.join(__dirname, "warm-images", `${group_id}`, `compressed_${id}.jpg`);
        await fs.writeFile(localCompressedPath, compressedBuffer);

        let downloadURLStripped;
        if (planType == 'elite') {
            const strippedPath = `stripped_${id}`;
            await bucket.file(strippedPath).save(strippedBuffer, {
                contentType: "image/jpeg",
                metadata: {
                    cacheControl: "public, max-age=31536000, immutable",
                    metadata: {
                        id: id,
                        filename: filename,
                        group_id: group_id,
                        user_id: created_by_user,
                        uploaded_at: uploaded_at,
                        source: 'google_drive',
                        drive_file_id: drive_path
                    }
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
                compressedBuffer3k = strippedBuffer;
                console.log(`✅ Image ${id} is ${originalWidth}px wide(≤ 3000px), using original size`);
            } else {
                compressedBuffer3k = await baseImage.rotate().resize({ width: 3000 }).jpeg().toBuffer();
                console.log(`✅ Resized Image ${id} from ${originalWidth}px to 3000px`);
            }
            const compressedPath3k = `compressed_3k_${id}`;
            await bucket.file(compressedPath3k).save(compressedBuffer3k, {
                contentType: 'image/jpeg',
                metadata: {
                    cacheControl: "public, max-age=31536000, immutable",
                    metadata: {
                        id: id,
                        filename: filename,
                        group_id: group_id,
                        user_id: created_by_user,
                        uploaded_at: uploaded_at,
                        source: 'google_drive',
                        drive_file_id: drive_path
                    }
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
                cacheControl: "public, max-age=31536000, immutable",
                metadata: {
                    id: id,
                    filename: filename,
                    group_id: group_id,
                    user_id: created_by_user,
                    uploaded_at: uploaded_at,
                    source: 'google_drive',
                    drive_file_id: drive_path
                }
            },
        });
        console.log(`✅ Created 200px thumbnail for Image ${id}`);

        const thumbFile = bucket.file(thumbPath);
        const compressedFile = bucket.file(compressedPath);

        // Get signed URLs
        const [downloadURL] = await thumbFile.getSignedUrl({
            action: 'read',
            expires: '03-09-2491'
        });

        const [downloadURLCompressed] = await compressedFile.getSignedUrl({
            action: 'read',
            expires: '03-09-2491'
        });

        console.log(`✅ Signed URLs generated for image ${id}`);

        return {
            id,
            success: true,
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
                signedUrlStripped: downloadURLStripped,
                file_size: image.file_size
            }
        };
    } catch (error) {
        console.error(`❌ Error processing Image ${id}: `, error.message);
        return {
            id,
            success: false,
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
                file_size: image.file_size
            }
        };
    }
}

// Modified function to process images from Drive
async function processImagesBatch(images, planType, parallel_limit) {
    console.log(`🔃 Processing batch of ${images.length} images with ${parallel_limit} parallel workers`);

    const results = [];

    for (let i = 0; i < images.length; i += parallel_limit) {
        const chunk = images.slice(i, i + parallel_limit);
        console.log(`🔃 Processing chunk ${Math.floor(i / parallel_limit) + 1} /${Math.ceil(images.length / parallel_limit)
            } (${chunk.length} images)`);

        const chunkPromises = chunk.map((image) => processSingleImageFromDrive(image, planType));
        const chunkResults = await Promise.all(chunkPromises);
        results.push(...chunkResults);

        console.log(`✅ Completed chunk ${Math.floor(i / parallel_limit) + 1} /${Math.ceil(images.length / parallel_limit)}`);
    }

    return results;
}

// Function to mark folder as processed
async function markFolderAsProcessed(client, driveFolderId) {
    try {
        await client.query(
            'UPDATE drive_folders SET is_processed = true, processed_time = NOW() WHERE id = $1',
            [driveFolderId]
        );
        console.log(`✅ Marked folder ${driveFolderId} as processed`);
        return { success: true };
    } catch (error) {
        console.error(`❌ Error marking folder ${driveFolderId} as processed:`, error.message);
        return { success: false, error: error.message };
    }
}

// Rest of the functions remain the same (getDynamicParallelLimit, insertIntoDatabaseBatch, etc.)
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

// [Include all other existing functions like performBatchInsert, performChunkedInserts, 
//  insertIntoDatabaseBatch, processImagesBatches, getGroupPlanType, updateStatusHistory, 
//  updateStatus, etc. - they remain unchanged]

// ... (include all the other existing functions here - they don't need modification)
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
// Modified main processing function
async function processImages() {
    const client = await pool.connect();

    try {
        while (true) {
            const run_id = Date.now()
            await updateStatusHistory(client, run_id, "node_compression", "run", null, null, null, null, "")
            console.log("🔃 Starting new processing cycle");

            let offset = 0;
            let hasMoreFolders = true;
            let totalProcessedAllBatches = 0;
            let totalCleanedAllBatches = 0;
            let totalBatchesProcessed = 0;
            let totalAllImagesProcessed = [];
            let totalAllImagesProcessFailed = [];
            let totalImagesInitialized = 0;
            let totalFailedFolders = [];

            // Keep fetching and processing folders until no more folders
            while (hasMoreFolders) {
                console.log(`🔃 Fetching batch ${totalBatchesProcessed + 1} from drive_folders table`);

                // Fetch batch of unprocessed folders from database
                const resFromFolderFetch = await fetchUnprocessedFoldersBatch(client, FIREBASE_BATCH_SIZE, offset);

                if (!resFromFolderFetch.success) {
                    throw new ProcessingError(resFromFolderFetch.error, {
                        reason: resFromFolderFetch.errorReason + " : " + resFromFolderFetch.error,
                        retryable: true,
                    });
                }

                const { folders, hasMore } = resFromFolderFetch;
                if (folders.length === 0) {
                    console.log("⏸️ No more unprocessed folders found in this batch");
                    hasMoreFolders = false;
                    break;
                }

                console.log(`✅ Batch ${totalBatchesProcessed + 1}: Found ${folders.length} folders`);

                let batchProcessedCount = 0;
                let batchCleanedCount = 0;

                // Process each folder in this batch
                for (const folder of folders) {
                    const { images, group_id, drive_folder_id } = folder;

                    if (images.length === 0) {
                        console.log(`⏸️ No images found in folder ${folder.folder_id} for group ${group_id}`);
                        // Mark folder as processed even if no images
                        await markFolderAsProcessed(client, drive_folder_id);
                        continue;
                    }

                    totalImagesInitialized += images.length;

                    await fs.mkdir(path.join(__dirname, "warm-images", `${group_id}`), { recursive: true });
                    console.log(`🔃 Processing ${images.length} images for group ${group_id} from folder ${folder.folder_id}`);

                    // Get plan type for this group
                    const planTypesResonse = await getGroupPlanType(client, [group_id]);
                    if (!planTypesResonse.success) {
                        throw new ProcessingError(planTypesResonse.error, {
                            groupId: group_id,
                            reason: planTypesResonse.errorReason + " : " + planTypesResonse.error,
                            retryable: false,
                        });
                    }

                    const planTypeEntry = planTypesResonse.planDeatils.find(p => p.groupId == group_id);
                    const planType = planTypeEntry?.planType;
                    console.log(`📋 Group ${group_id} plan type: ${planType}`);

                    if (!["lite", "elite", "pro"].includes(planType)) {
                        await updateStatusHistory(client, run_id, "node_compression", "group", images.length, images.length, 0, group_id, "Plan Type not found for group " + group_id);
                        totalFailedFolders.push(folder.folder_id);
                        continue;
                    }

                    try {
                        // Process all images in this folder
                        const { totalProcessed, totalCleaned, allResults, totalImagesInsertedIntoDB, totalImagesFailedDBInsertion } =
                            await processImagesBatches(client, images, planType, group_id, run_id);

                        totalAllImagesProcessed.push(...totalImagesInsertedIntoDB);
                        totalAllImagesProcessFailed.push(...totalImagesFailedDBInsertion);

                        await updateStatusHistory(client, run_id, "node_compression", "group", images.length,
                            totalImagesFailedDBInsertion.length, totalImagesInsertedIntoDB.length, group_id,
                            "DB insertion failed for " + totalImagesFailedDBInsertion.join(" , \n") + " \n");

                        batchProcessedCount += totalProcessed;
                        batchCleanedCount += totalCleaned;

                        // Mark folder as processed after successful processing
                        await markFolderAsProcessed(client, drive_folder_id);

                        console.log(`✅ Completed processing for group ${group_id}, folder ${folder.folder_id}. Processed: ${totalProcessed}/${images.length}, Cleaned: ${totalCleaned}`);

                    } catch (folderError) {
                        console.error(`❌ Error processing folder ${folder.folder_id}:`, folderError.message);
                        totalFailedFolders.push(folder.folder_id);
                        await updateStatusHistory(client, run_id, "node_compression", "folder_error", images.length, images.length, 0, group_id,
                            "Folder processing failed: " + folderError.message);
                    }
                }

                totalProcessedAllBatches += batchProcessedCount;
                totalCleanedAllBatches += batchCleanedCount;
                totalBatchesProcessed++;

                console.log(`🎉 Finished processing batch ${totalBatchesProcessed}. Batch processed: ${batchProcessedCount}, Batch cleaned: ${batchCleanedCount}, Total processed: ${totalProcessedAllBatches}, Total cleaned: ${totalCleanedAllBatches}`);

                // Update pagination state
                offset += FIREBASE_BATCH_SIZE;
                hasMoreFolders = hasMore;

                // Brief pause between batches
                if (hasMoreFolders) {
                    console.log(`⏸️ Brief pause before fetching next batch...`);
                    await new Promise(resolve => setTimeout(resolve, 1000));
                }
            }

            if (totalProcessedAllBatches === 0) {
                console.log("⏸️ No unprocessed folders found in entire cycle, waiting...");
                await updateStatusHistory(client, run_id, "node_compression", "run", totalImagesInitialized,
                    totalImagesInitialized - totalAllImagesProcessed.length, totalAllImagesProcessed.length, null,
                    "DB insertion failed for " + totalAllImagesProcessFailed.join(" , \n") + " \n" +
                    "Failed Folders:" + totalFailedFolders.join(" , \n"));
                await new Promise(res => setTimeout(res, 300000)); // wait 5 minutes
            } else {
                console.log(`🎉 Completed full processing cycle. Total batches: ${totalBatchesProcessed}, Total processed: ${totalProcessedAllBatches}, Total files cleaned up: ${totalCleanedAllBatches}`);

                console.log(`⏸️ Brief pause before starting next cycle...`);
                await new Promise(resolve => setTimeout(resolve, 5000));
            }
        }
    } catch (err) {
        if (err instanceof ProcessingError) {
            console.error(`❌ ProcessingError: ${err.message}`, {
                groupId: err.groupId,
                reason: err.reason,
            });

            await updateStatus(client, null, err.reason, true);

        } else {
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
async function performBatchInsert(client, successfulResults) {
    try {
        const valuesClauses = [];
        const allParams = [];
        let paramIndex = 1;

        for (const result of successfulResults) {
            const { data } = result;

            valuesClauses.push(
                `($${paramIndex}, $${paramIndex + 1}, $${paramIndex + 2}, $${paramIndex + 3}, $${paramIndex + 4}::timestamp, $${paramIndex + 5}, $${paramIndex + 6}::jsonb, $${paramIndex + 7}::bytea, $${paramIndex + 8}::bytea, $${paramIndex + 9}, $${paramIndex + 10}, $${paramIndex + 11}::timestamp, $${paramIndex + 12}, $${paramIndex + 13}, $${paramIndex + 14} , $${paramIndex + 15}, $${paramIndex + 16} , NOW())`
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
                data.signedUrlStripped,
                data.file_size
            );

            paramIndex += 17;
        }

        const batchInsertQuery = `
        INSERT INTO images_drive
        (id, group_id, created_by_user, filename, uploaded_at, status, json_meta_data, thumb_byte, image_byte, compressed_location, artist, date_taken, location, signed_url, signed_url_3k , signed_url_stripped ,size, last_processed_at)
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
            signed_url_stripped = EXCLUDED.signed_url_stripped,
            size = EXCLUDED.size
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

        // for (const id of insertionResult.failedIds) {
        //     try {
        //         const res = await renameFailedFiles(id)
        //         if (!res.success) {
        //             await updateStatusHistory(client, run_id, "node_compression", "failed_renaming", insertionResult.failedIds.length, null, null, groupId, "Failed Files cannot be renamed")
        //             throw new ProcessingError("Cannot rename failed files", {
        //                 groupId: groupId,
        //                 reason: "Cannot rename failed files : " + res.error
        //             })
        //         }

        //     } catch (error) {
        //         await updateStatusHistory(client, run_id, "node_compression", "failed_renaming", insertionResult.failedIds.length, null, null, groupId, "Failed Files cannot be renamed")
        //         throw new ProcessingError("Cannot rename failed files", {
        //             groupId: groupId,
        //             reason: "Cannot rename failed files : " + error.message
        //         })
        //     }
        // }


        // for (const record of batchResults.filter(r => !r.success)) {
        //     try {
        //         const res = await renameFailedFiles(record.id)
        //         if (!res.success) {
        //             await updateStatusHistory(client, run_id, "node_compression", "failed_renaming", insertionResult.failedIds.length, null, null, groupId, "Failed Files cannot be renamed")
        //             throw new ProcessingError("Cannot rename failed files", {
        //                 groupId: groupId,
        //                 reason: "Cannot rename failed files : " + res.error
        //             })
        //         }
        //     } catch (error) {
        //         await updateStatusHistory(client, run_id, "node_compression", "failed_renaming", insertionResult.failedIds.length, null, null, groupId, "Failed Files cannot be renamed")
        //         throw new ProcessingError("Cannot rename failed files", {
        //             groupId: groupId,
        //             reason: "Cannot rename failed files : " + error.message
        //         })
        //     }
        // }
        // Only cleanup if database insertion was successful
        // if (insertionResult.success) {
        //     console.log(`🔃 Cleaning up original files for batch ${batchNumber}/${totalBatches}`);

        // const cleanupResult = await cleanupOriginalFiles(insertionResult.insertedIds);

        // totalCleaned += cleanupResult.totalDeleted;
        // totalCleanupFailed += cleanupResult.totalFailed;

        // if (!cleanupResult.cleanupSuccess) {
        //     console.warn(`⚠️ Some files in batch ${batchNumber}/${totalBatches} could not be cleaned up`);
        //     await updateStatusHistory(client, run_id, "node_compression", "cleaning", insertionResult.insertedIds.length, null, null, groupId, "Some files in batch could not be cleaned up")
        // }
        // }
        if (!insertionResult.success || insertionResult.failedIds.length > 0) {
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
// Run the main process if this file is executed directly
if (require.main === module) {
    processImages();
}