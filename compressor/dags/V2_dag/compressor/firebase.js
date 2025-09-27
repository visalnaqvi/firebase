const { Pool } = require('pg');
const admin = require('firebase-admin');

const { join } = require('path');
const serviceAccount = require('../firebase-key.json');

const fs = require('fs').promises;

// your local modules
const getGroupPlanType = require('./shared/getGroupPlanType');
const processImagesBatch = require('./shared/processImagesBatch');
const insertIntoDatabaseBatch = require('./shared/insertIntoDatabaseBatch');
admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
    // DEV BUCKET
    storageBucket: 'gallery-585ee.firebasestorage.app',
    // PROD BUCKET
    // storageBucket: 'gallery-585ee-production',
});
const bucket = admin.storage().bucket();

const pool = new Pool({
    connectionString: "postgresql://postgres:AfldldzckDWtkskkAMEhMaDXnMqknaPY@ballast.proxy.rlwy.net:56193/railway"
    // connectionString: "postgresql://postgres:kdVrNTrtLzzAaOXzKHaJCzhmoHnSDKDG@nozomi.proxy.rlwy.net:24794/railway"
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
const BATCH_SIZE = 10;
const PARALLEL_LIMIT = 10;
const FIREBASE_BATCH_SIZE = 50;
const CLEANUP_BATCH_SIZE = 20;
const MAX_CLEANUP_RETRIES = 3;

// Your existing Firebase functions (keeping them exactly as they are)
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

        // First Pass
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
                    failedImages.push(file);
                    continue;
                }

                unprocessedImages.push(imageData);
            } catch (error) {
                console.error(`❌ Error reading metadata for ${file.name}:`, error.message);
                failedImages.push(file);
            }
        }

        // Second Retry Pass
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
        const batchResults = await processImagesBatch(batch, planType, parallelLimit, bucket, "firebase", null);
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
// Main processing function for Firebase (simplified)
async function processFirebaseImages() {
    const client = await pool.connect();

    try {
        console.log("🔥 Starting Firebase image processing");

        let pageToken = null;
        let hasMoreImages = true;
        let totalProcessed = 0;

        while (hasMoreImages) {
            const resFromFirebaseFetch = await fetchUnprocessedImagesBatch(FIREBASE_BATCH_SIZE, pageToken);

            if (!resFromFirebaseFetch.success) {
                throw new ProcessingError(resFromFirebaseFetch.error, {
                    reason: resFromFirebaseFetch.errorReason,
                    retryable: true,
                });
            }

            const { images: unprocessedImages, nextPageToken, hasMore } = resFromFirebaseFetch;

            if (unprocessedImages.length === 0) {
                console.log("⏸️ No more unprocessed images found");
                break;
            }

            // Group images by group_id and process them
            const imagesByGroup = {};
            for (const image of unprocessedImages) {
                if (!imagesByGroup[image.group_id]) {
                    imagesByGroup[image.group_id] = [];
                }
                imagesByGroup[image.group_id].push(image);
            }

            // Process each group
            for (const [groupId, groupImages] of Object.entries(imagesByGroup)) {
                await fs.mkdir(join(__dirname, "..", "warm-images", `${groupId}`), { recursive: true });

                // Get plan type and process images
                const planTypesResponse = await getGroupPlanType(client, [groupId]);
                if (!planTypesResponse.success) continue;

                const planTypeEntry = planTypesResponse.planDetails.find(p => p.groupId == groupId);
                const planType = planTypeEntry?.planType;

                if (!["lite", "elite", "pro"].includes(planType)) continue;

                // Process images for this group
                const { totalImagesInsertedIntoDB } = await processImagesBatches(client, groupImages, planType, groupId, Date.now());
                totalProcessed += totalImagesInsertedIntoDB.length;
            }

            pageToken = nextPageToken;
            hasMoreImages = hasMore;
        }

        console.log(`🎉 Firebase processing completed. Total processed: ${totalProcessed}`);
        return { success: true, totalProcessed };

    } catch (error) {
        console.error('❌ Firebase processing failed:', error);
        return { success: false, error: error.message };
    } finally {
        client.release();
    }
}

// Run if this file is executed directly
if (require.main === module) {
    processFirebaseImages()
        .then(result => {
            if (result.success) {
                console.log('✅ Firebase processing completed successfully');
                process.exit(0);
            } else {
                console.error('❌ Firebase processing failed');
                process.exit(1);
            }
        })
        .catch(error => {
            console.error('❌ Unexpected error:', error);
            process.exit(1);
        });
}

module.exports = processFirebaseImages;