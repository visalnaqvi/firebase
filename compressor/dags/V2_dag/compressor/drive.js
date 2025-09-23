const { Pool } = require('pg');
const { join } = require('path');
const fs = require('fs').promises;
const { google } = require('googleapis');
const { post } = require('axios');
const admin = require('firebase-admin');
const serviceAccount = require('../firebase-key.json');
const getGroupPlanType = require('./shared/getGroupPlanType');
const processImagesBatch = require('./shared/processImagesBatch');
const insertIntoDatabaseBatch = require('./shared/insertIntoDatabaseBatch');

const pool = new Pool({
    connectionString: "postgresql://postgres:AfldldzckDWtkskkAMEhMaDXnMqknaPY@ballast.proxy.rlwy.net:56193/railway"
});
admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
    // DEV BUCKET
    storageBucket: 'gallery-585ee.firebasestorage.app',
    // PROD BUCKET
    // storageBucket: 'gallery-585ee-production',
});
const bucket = admin.storage().bucket();
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
const BATCH_SIZE = 10;
const PARALLEL_LIMIT = 10;
const FIREBASE_BATCH_SIZE = 50;

// Google Drive API helper functions
async function refreshAccessToken(refreshToken, clientId, clientSecret) {
    try {
        const response = await post('https://oauth2.googleapis.com/token', {
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
            id,
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
        const auth = new google.auth.OAuth2();
        auth.setCredentials({ access_token: accessToken });

        const response = await drive.files.list({
            auth: auth,
            q: `'${folderId}' in parents and mimeType contains 'image/' and trashed=false`,
            fields: 'files(id,name,size,mimeType,createdTime,modifiedTime)',
            pageSize: 1000
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

async function fetchUnprocessedFoldersBatch(client, maxResults = FIREBASE_BATCH_SIZE, offset = 0) {
    console.log(`🔃 Fetching batch of up to ${maxResults} unprocessed folders from drive_folders table`);

    try {
        const { rows: folders } = await client.query(
            `SELECT df.folder_id, df.group_id, df.user_id, df.id as drive_folder_id
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
                const tokenResult = await getValidAccessToken(client, folder.user_id, folder.group_id);
                if (!tokenResult.success) {
                    console.error(`❌ Cannot get access token for user ${folder.user_id}: ${tokenResult.error}`);
                    continue;
                }

                const filesResult = await listDriveFiles(tokenResult.accessToken, folder.folder_id);
                if (!filesResult.success) {
                    console.error(`❌ Cannot list files in folder ${folder.folder_id}: ${filesResult.error}`);
                    continue;
                }

                const images = filesResult.files.map(file => ({
                    id: file.id,
                    filename: file.name,
                    group_id: folder.group_id,
                    created_by_user: folder.user_id,
                    file_size: parseInt(file.size) || 0,
                    content_type: file.mimeType,
                    uploaded_at: file.createdTime,
                    folder_id: folder.folder_id,
                    drive_folder_id: folder.drive_folder_id,
                    access_token: tokenResult.accessToken,
                    drive_path: file.id // Use Drive file ID as the path
                }));

                processedFolders.push({
                    ...folder,
                    images: images,
                    access_token: tokenResult.accessToken
                });

            } catch (error) {
                console.error(`❌ Error processing folder ${folder.folder_id}:`, error.message);
            }
        }

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

async function processImagesBatches(client, images, planType, groupId, run_id) {
    console.log(`🔃 Processing ${images.length} images in batches of ${BATCH_SIZE}`);
    let allResults = [];
    let totalImagesInsertedIntoDB = []
    let totalImagesFailedDBInsertion = []

    for (let i = 0; i < images.length; i += BATCH_SIZE) {
        const batch = images.slice(i, i + BATCH_SIZE);
        const batchNumber = Math.floor(i / BATCH_SIZE) + 1;
        const totalBatches = Math.ceil(images.length / BATCH_SIZE);

        console.log(`🔃 Processing batch ${batchNumber}/${totalBatches} (${batch.length} images)`);

        const batchResults = await processImagesBatch(batch, planType, PARALLEL_LIMIT, bucket, "drive");
        allResults.push(...batchResults);

        console.log(`🔃 Inserting database for batch ${batchNumber}/${totalBatches}`);
        const insertionResult = await insertIntoDatabaseBatch(client, batchResults, run_id);

        totalImagesInsertedIntoDB.push(...insertionResult.insertedIds);
        totalImagesFailedDBInsertion.push(...insertionResult.failedIds);
        totalImagesFailedDBInsertion.push(...batchResults.filter(r => !r.success));

        if (!insertionResult.success) {
            throw new ProcessingError("Failed to do database insert", {
                reason: "Failed to do database insert : " + insertionResult.error
            })
        }

        console.log(`✅ Completed batch ${batchNumber}/${totalBatches}`);
    }

    return {
        totalCleaned: 0, // No cleanup needed for Drive
        totalCleanupFailed: 0,
        allResults,
        totalImagesInsertedIntoDB,
        totalImagesFailedDBInsertion
    };
}

// Main processing function for Drive
async function processDriveImages() {
    const client = await pool.connect();

    try {
        console.log("🚗 Starting Drive image processing");

        let offset = 0;
        let hasMoreFolders = true;
        let totalProcessed = 0;

        while (hasMoreFolders) {
            const resFromFolderFetch = await fetchUnprocessedFoldersBatch(client, FIREBASE_BATCH_SIZE, offset);

            if (!resFromFolderFetch.success) {
                throw new ProcessingError(resFromFolderFetch.error, {
                    reason: resFromFolderFetch.errorReason,
                    retryable: true,
                });
            }

            const { folders, hasMore } = resFromFolderFetch;

            if (folders.length === 0) {
                console.log("⏸️ No more unprocessed folders found");
                break;
            }

            // Process each folder
            for (const folder of folders) {
                const { images, group_id, drive_folder_id } = folder;

                if (images.length === 0) {
                    console.log(`⏸️ No images found in folder ${folder.folder_id} for group ${group_id}`);
                    await markFolderAsProcessed(client, drive_folder_id);
                    continue;
                }

                await fs.mkdir(join(__dirname, "..", "warm-images", `${group_id}`), { recursive: true });
                console.log(`🔃 Processing ${images.length} images for group ${group_id} from folder ${folder.folder_id}`);

                // Get plan type for this group
                const planTypesResponse = await getGroupPlanType(client, [group_id]);
                if (!planTypesResponse.success) continue;

                const planTypeEntry = planTypesResponse.planDetails.find(p => p.groupId == group_id);
                const planType = planTypeEntry?.planType;

                if (!["lite", "elite", "pro"].includes(planType)) continue;

                try {
                    const { totalImagesInsertedIntoDB } = await processImagesBatches(client, images, planType, group_id, Date.now());
                    totalProcessed += totalImagesInsertedIntoDB.length;

                    // Mark folder as processed after successful processing
                    await markFolderAsProcessed(client, drive_folder_id);

                    console.log(`✅ Completed processing for group ${group_id}, folder ${folder.folder_id}. Processed: ${totalImagesInsertedIntoDB.length}/${images.length}`);

                } catch (folderError) {
                    console.error(`❌ Error processing folder ${folder.folder_id}:`, folderError.message);
                }
            }

            offset += FIREBASE_BATCH_SIZE;
            hasMoreFolders = hasMore;
        }

        console.log(`🎉 Drive processing completed. Total processed: ${totalProcessed}`);
        return { success: true, totalProcessed };

    } catch (error) {
        console.error('❌ Drive processing failed:', error);
        return { success: false, error: error.message };
    } finally {
        client.release();
    }
}

// Run if this file is executed directly
if (require.main === module) {
    processDriveImages()
        .then(result => {
            if (result.success) {
                console.log('✅ Drive processing completed successfully');
                process.exit(0);
            } else {
                console.error('❌ Drive processing failed');
                process.exit(1);
            }
        })
        .catch(error => {
            console.error('❌ Unexpected error:', error);
            process.exit(1);
        });
}

module.exports = processDriveImages;