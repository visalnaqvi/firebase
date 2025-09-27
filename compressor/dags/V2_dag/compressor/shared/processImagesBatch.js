const { google } = require('googleapis');
const sharp = require('sharp');
const exifParser = require('exif-parser');
const fs = require('fs').promises;
const path = require('path');


async function downloadFileFromDrive(drive, fileId) {
    try {
        const res = await drive.files.get(
            { fileId, alt: "media" },
            { responseType: "stream" }
        );

        const chunks = [];
        for await (const chunk of res.data) {
            chunks.push(chunk);
        }

        return { success: true, buffer: Buffer.concat(chunks) };
    } catch (error) {
        console.error("Error downloading file from Drive:", error.message);
        return { success: false, error: error.message };
    }
}

async function processSingleImage(image, planType, bucket, source, drive) {
    const { id, firebase_path, filename, group_id, created_by_user, uploaded_at } = image;

    console.log(`🔃 Processing Image ${id} for group ${group_id}`);

    try {
        console.log(`🔧 Processing image ${id} from ${firebase_path}`);
        let originalBuffer
        if (source == "firebase") {
            // Download image from Firebase
            const [buffer] = await bucket.file(firebase_path).download();
            originalBuffer = buffer

            console.log(`✅ Downloaded Image ${id} from Firebase`);
        } else if (source == "drive") {
            const downloadResult = await downloadFileFromDrive(drive, id);
            if (!downloadResult.success) {
                throw new Error(`Failed to download from Drive: ${downloadResult.error}`);
            }

            originalBuffer = downloadResult.buffer;
            console.log(`✅ Downloaded Image ${id} from Google Drive`);
        }


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

        const compressedPath = `compressed_${id}`;
        await bucket.file(compressedPath).save(compressedBuffer, {
            contentType: 'image/jpeg',
            metadata: {
                cacheControl: "public, max-age=31536000, immutable"
            },
        });

        const localCompressedPath = path.join(__dirname, "..", "..", "warm-images", `${group_id}`, `compressed_${id}.jpg`);
        await fs.writeFile(localCompressedPath, compressedBuffer);

        let downloadURLStripped;
        if (planType == 'elite') {
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
            firebase_path,
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
            firebase_path,
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


async function processImagesBatch(images, planType, parallel_limit, bucket, source, drive) {
    console.log(`🔃 Processing batch of ${images.length} images with ${parallel_limit} parallel workers`);

    const results = [];

    // Process images in chunks of PARALLEL_LIMIT
    for (let i = 0; i < images.length; i += parallel_limit) {
        const chunk = images.slice(i, i + parallel_limit);
        console.log(`🔃 Processing chunk ${Math.floor(i / parallel_limit) + 1} /${Math.ceil(images.length / parallel_limit)
            } (${chunk.length} images)`);

        const chunkPromises = chunk.map((image) => processSingleImage(image, planType, bucket, source, drive));
        const chunkResults = await Promise.all(chunkPromises);
        results.push(...chunkResults);

        console.log(`✅ Completed chunk ${Math.floor(i / parallel_limit) + 1} /${Math.ceil(images.length / parallel_limit)}`);
    }

    return results;
}

module.exports = processImagesBatch