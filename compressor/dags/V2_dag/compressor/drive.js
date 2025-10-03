const { Pool } = require('pg');
const { google } = require('googleapis');
const admin = require('firebase-admin');
const { join } = require('path');
const fs = require('fs').promises;

// Import your existing modules
const getGroupPlanType = require('./shared/getGroupPlanType');
const processImagesBatch = require('./shared/processImagesBatch');
const insertIntoDatabaseBatch = require('./shared/insertIntoDatabaseBatch');

// Firebase setup (using same config as your Firebase script)
const serviceAccount = require('../firebase-key.json');

admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
    // DEV BUCKET
    storageBucket: 'gallery-585ee.firebasestorage.app',
    // PROD BUCKET
    // storageBucket: 'gallery-585ee-production',
});
const bucket = admin.storage().bucket();

// Database connection
const pool = new Pool({
    connectionString: "postgresql://postgres:AfldldzckDWtkskkAMEhMaDXnMqknaPY@ballast.proxy.rlwy.net:56193/railway"
    // connectionString: "postgresql://postgres:kdVrNTrtLzzAaOXzKHaJCzhmoHnSDKDG@nozomi.proxy.rlwy.net:24794/railway"
});

// Google Drive setup
const KEYFILEPATH = "./snapper-gallery-75b9ac194113.json";
const SCOPES = ["https://www.googleapis.com/auth/drive.readonly"];

const auth = new google.auth.GoogleAuth({
    keyFile: KEYFILEPATH,
    scopes: SCOPES,
});

const drive = google.drive({ version: "v3", auth });

// Configuration
const BATCH_SIZE = 10;
const PARALLEL_LIMIT = 10;
const FOLDER_BATCH_SIZE = 5;

class ProcessingError extends Error {
    constructor(message, { groupId = null, reason = null, retryable = true } = {}) {
        super(message);
        this.name = "ProcessingError";
        this.groupId = groupId;
        this.reason = reason;
        this.retryable = retryable;
    }
}

// Fetch unprocessed folders from database
async function fetchUnprocessedFolders(client, limit = FOLDER_BATCH_SIZE, offset = 0) {
    console.log(`🔃 Fetching ${limit} unprocessed folders from database (offset: ${offset})`);

    try {
        const query = `
            SELECT df.folder_id, df.group_id, df.user_id, df.id   
            FROM drive_folders df               
            WHERE df.is_processed = false               
            ORDER BY df.created_at               
            LIMIT $1 OFFSET $2
        `;

        const result = await client.query(query, [limit, offset]);

        console.log(`✅ Found ${result.rows.length} unprocessed folders`);

        return {
            success: true,
            folders: result.rows,
            hasMore: result.rows.length === limit
        };
    } catch (error) {
        console.error('❌ Error fetching folders from database:', error.message);
        return {
            success: false,
            error: error.message,
            folders: [],
            hasMore: false
        };
    }
}

// List all image files in a Google Drive folder with comprehensive validation
// List all image files in a Google Drive folder with comprehensive validation and pagination
async function listImagesInDriveFolder(folderId) {
    console.log(`🔃 Listing images in Drive folder: ${folderId}`);

    try {
        // Use broader query to get all files, then filter
        const query = `'${folderId}' in parents and mimeType contains 'image/' and trashed=false`;

        let allFiles = [];
        let pageToken = null;
        let pageCount = 0;

        do {
            pageCount++;
            console.log(`📄 Fetching page ${pageCount} of images...`);

            const res = await drive.files.list({
                q: query,
                fields: "nextPageToken, files(id, name, mimeType, size, createdTime, modifiedTime)",
                pageSize: 1000,
                pageToken: pageToken || undefined
            });

            const currentPageFiles = res.data.files || [];
            allFiles = allFiles.concat(currentPageFiles);

            console.log(`📄 Page ${pageCount}: Found ${currentPageFiles.length} files (Total so far: ${allFiles.length})`);

            pageToken = res.data.nextPageToken;

        } while (pageToken);

        console.log(`📁 Total Files Retrieved across ${pageCount} pages: ${allFiles.length}`);

        // Filter out system files and suspicious files
        const filteredFiles = allFiles.filter(file => {
            // Skip files with suspicious names
            const suspiciousNames = [
                '.DS_Store',
                'Thumbs.db',
                'desktop.ini',
                '.localized',
                '__MACOSX',
                '.fseventsd',
                '.Spotlight-V100',
                '.Trashes',
                '.TemporaryItems'
            ];

            const fileName = file.name.toLowerCase();

            // Check for suspicious file names
            if (suspiciousNames.some(suspicious => fileName.includes(suspicious.toLowerCase()))) {
                console.log(`🚫 Skipping system file: ${file.name}`);
                return false;
            }

            // Check for valid image extensions
            const validExtensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif', '.svg'];
            const hasValidExtension = validExtensions.some(ext => fileName.endsWith(ext));

            if (!hasValidExtension) {
                console.log(`🚫 Skipping file without image extension: ${file.name}`);
                return false;
            }

            // Check for suspicious file sizes (exactly 4096 bytes is often a system file)
            if (parseInt(file.size) === 4096) {
                console.log(`🚫 Skipping suspicious 4096-byte file: ${file.name}`);
                return false;
            }

            // Skip very small files (likely not real images)
            if (parseInt(file.size) < 1024) { // Less than 1KB
                console.log(`🚫 Skipping tiny file: ${file.name} (${file.size} bytes)`);
                return false;
            }

            // Check for valid image MIME types more strictly
            const validMimeTypes = [
                'image/jpeg',
                'image/jpg',
                'image/png',
                'image/gif',
                'image/bmp',
                'image/webp',
                'image/tiff',
                'image/svg+xml'
            ];

            if (!validMimeTypes.includes(file.mimeType.toLowerCase())) {
                console.log(`🚫 Skipping file with invalid MIME type: ${file.name} (${file.mimeType})`);
                return false;
            }

            return true;
        });

        console.log(`📊 Filtered ${allFiles.length} files down to ${filteredFiles.length} valid images`);
        console.log(`📊 Processing completed across ${pageCount} pages`);

        return {
            success: true,
            files: filteredFiles,
            totalFound: allFiles.length,
            validImages: filteredFiles.length,
            pagesProcessed: pageCount
        };
    } catch (error) {
        console.error(`❌ Error listing files in Drive folder ${folderId}:`, error.message);
        return {
            success: false,
            error: error.message,
            files: [],
            totalFound: 0,
            validImages: 0,
            pagesProcessed: 0
        };
    }
}

// Convert Drive file metadata to image format for processing
function convertDriveFileToImageFormat(file, groupId, userId) {
    return {
        id: file.id,
        filename: file.name,
        group_id: groupId,
        created_by_user: userId,
        firebase_path: null, // Not applicable for Drive
        file_size: parseInt(file.size) || 0,
        content_type: file.mimeType,
        uploaded_at: file.createdTime,
        access_token: null // Not needed with service account
    };
}

// Process images from Drive folder
async function processDriveFolderImages(client, folder, planType, run_id) {
    const { folder_id: folderId, group_id: groupId, user_id: userId } = folder;

    console.log(`🔃 Processing Drive folder ${folderId} for group ${groupId}`);

    try {
        // Create local directory for group
        await fs.mkdir(join(__dirname, "..", "warm-images", `${groupId}`), { recursive: true });

        // List images in the Drive folder
        const listResult = await listImagesInDriveFolder(folderId);

        if (!listResult.success) {
            throw new ProcessingError(`Failed to list images in folder: ${listResult.error}`, {
                groupId: groupId,
                reason: listResult.error
            });
        }

        const driveFiles = listResult.files;

        if (driveFiles.length === 0) {
            console.log(`⚠️ No images found in Drive folder ${folderId}`);
            return {
                totalImagesInsertedIntoDB: [],
                totalImagesFailedDBInsertion: [],
                allResults: []
            };
        }

        // Convert Drive files to image format
        const images = driveFiles.map(file =>
            convertDriveFileToImageFormat(file, groupId, userId)
        );

        console.log(`🔃 Processing ${images.length} images from Drive folder ${folderId}`);

        // Process images in batches
        let allResults = [];
        let totalImagesInsertedIntoDB = [];
        let totalImagesFailedDBInsertion = [];

        for (let i = 0; i < images.length; i += BATCH_SIZE) {
            const batch = images.slice(i, i + BATCH_SIZE);
            const batchNumber = Math.floor(i / BATCH_SIZE) + 1;
            const totalBatches = Math.ceil(images.length / BATCH_SIZE);

            console.log(`🔃 Processing batch ${batchNumber}/${totalBatches} (${batch.length} images)`);

            // Process this batch using your existing function
            const batchResults = await processImagesBatch(batch, planType, PARALLEL_LIMIT, bucket, "drive", drive);
            allResults.push(...batchResults);

            // Insert into database
            console.log(`🔃 Inserting database records for batch ${batchNumber}/${totalBatches}`);
            const insertionResult = await insertIntoDatabaseBatch(client, batchResults, run_id);

            totalImagesInsertedIntoDB.push(...insertionResult.insertedIds);
            totalImagesFailedDBInsertion.push(...insertionResult.failedIds);
            totalImagesFailedDBInsertion.push(...batchResults.filter(r => !r.success));

            if (!insertionResult.success) {
                throw new ProcessingError("Failed to insert into database", {
                    groupId: groupId,
                    reason: "Failed to insert batch into database: " + insertionResult.error
                });
            }

            console.log(`✅ Completed batch ${batchNumber}/${totalBatches}`);

            // Brief pause between batches
            if (batchNumber < totalBatches) {
                console.log(`⏸️  Brief pause before next batch...`);
                await new Promise(resolve => setTimeout(resolve, 100));
            }
        }

        return {
            totalImagesInsertedIntoDB,
            totalImagesFailedDBInsertion,
            allResults
        };

    } catch (error) {
        console.error(`❌ Error processing Drive folder ${folderId}:`, error.message);
        throw error;
    }
}

// Update folder processing status
async function updateFolderProcessingStatus(client, folderId, isProcessed = true) {
    console.log(`🔃 Updating folder ${folderId} processing status to: ${isProcessed}`);

    try {
        const query = `
            UPDATE drive_folders 
            SET is_processed = $1, processed_time = CURRENT_TIMESTAMP 
            WHERE id = $2
        `;

        await client.query(query, [isProcessed, folderId]);
        console.log(`✅ Updated folder ${folderId} processing status`);

        return { success: true };
    } catch (error) {
        console.error(`❌ Error updating folder ${folderId} status:`, error.message);
        return { success: false, error: error.message };
    }
}

// Main processing function for Google Drive
async function processDriveImages() {
    const client = await pool.connect();

    try {
        console.log("🚗 Starting Google Drive image processing");

        let offset = 0;
        let hasMoreFolders = true;
        let totalProcessed = 0;
        let totalFoldersProcessed = 0;

        while (hasMoreFolders) {
            // Fetch batch of unprocessed folders
            const foldersResult = await fetchUnprocessedFolders(client, FOLDER_BATCH_SIZE, offset);

            if (!foldersResult.success) {
                throw new ProcessingError(foldersResult.error, {
                    reason: "Failed to fetch folders from database",
                    retryable: true
                });
            }

            const { folders, hasMore } = foldersResult;

            if (folders.length === 0) {
                console.log("⏸️ No more unprocessed folders found");
                break;
            }

            // Process each folder
            for (const folder of folders) {
                try {
                    console.log(`🔃 Processing folder ${folder.folder_id} for group ${folder.group_id}`);

                    // Get plan type for the group
                    const planTypesResponse = await getGroupPlanType(client, [folder.group_id]);
                    if (!planTypesResponse.success) {
                        console.warn(`⚠️ Could not get plan type for group ${folder.group_id}, skipping`);
                        continue;
                    }

                    const planTypeEntry = planTypesResponse.planDetails.find(p => p.groupId == folder.group_id);
                    const planType = planTypeEntry?.planType;

                    if (!["lite", "elite", "pro"].includes(planType)) {
                        console.warn(`⚠️ Invalid plan type '${planType}' for group ${folder.group_id}, skipping`);
                        continue;
                    }

                    // Process images in this folder
                    const run_id = Date.now();
                    const processingResult = await processDriveFolderImages(client, folder, planType, run_id);

                    totalProcessed += processingResult.totalImagesInsertedIntoDB.length;

                    // Update folder status to processed
                    const updateResult = await updateFolderProcessingStatus(client, folder.id, true);

                    if (!updateResult.success) {
                        console.warn(`⚠️ Could not update processing status for folder ${folder.id}: ${updateResult.error}`);
                    }

                    totalFoldersProcessed++;

                    console.log(`✅ Completed processing folder ${folder.folder_id} - ${processingResult.totalImagesInsertedIntoDB.length} images processed`);

                } catch (error) {
                    console.error(`❌ Error processing folder ${folder.folder_id}:`, error.message);

                    // Mark folder as failed but don't stop processing other folders
                    try {
                        await updateFolderProcessingStatus(client, folder.id, false);
                    } catch (updateError) {
                        console.error(`❌ Could not update failed status for folder ${folder.id}:`, updateError.message);
                    }

                    continue; // Continue with next folder
                }
            }

            offset += FOLDER_BATCH_SIZE;
            hasMoreFolders = hasMore;
        }

        console.log(`🎉 Drive processing completed. Total folders processed: ${totalFoldersProcessed}, Total images processed: ${totalProcessed}`);
        return { success: true, totalFoldersProcessed, totalProcessed };

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
                console.log(`📊 Final stats: ${result.totalFoldersProcessed} folders, ${result.totalProcessed} images`);
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