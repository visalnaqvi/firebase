// Enhanced processSingleImage function with better format handling
async function processSingleImage(image, planType) {
    const { id, firebase_path, filename, group_id, created_by_user, uploaded_at } = image;

    console.log(`🔃 Processing Image ${id} for group ${group_id}`);

    try {
        console.log(`🔧 Processing image ${id} from ${firebase_path}`);

        // Download image from Firebase
        const [originalBuffer] = await bucket.file(firebase_path).download();
        console.log(`✅ Downloaded Image ${id} from Firebase`);

        // Create Sharp instance and get metadata
        let sharpInstance = sharp(originalBuffer);
        const sharpMeta = await sharpInstance.metadata();
        const originalWidth = sharpMeta.width;
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

        // Convert to JPEG and strip metadata with auto-rotation
        // Sharp automatically handles format conversion when you specify .jpeg()
        const baseProcessedBuffer = await sharpInstance
            .rotate() // Auto-rotate based on EXIF orientation
            .jpeg({ quality: 90 }) // Convert to JPEG with good quality
            .toBuffer();

        console.log(`✅ Converted and stripped metadata for Image ${id}`);

        // Create a new Sharp instance from the processed buffer for further operations
        const baseImage = sharp(baseProcessedBuffer);
        let compressedBuffer;
        let compressedBuffer3k;

        // Generate 1000px version
        if (originalWidth <= 1000) {
            // Use processed buffer as compressed since it's already <= 1000px
            compressedBuffer = baseProcessedBuffer;
            console.log(`✅ Image ${id} is ${originalWidth}px wide (≤ 1000px), using original size`);
        } else {
            // Resize to 1000px
            compressedBuffer = await baseImage
                .resize({ width: 1000 })
                .jpeg({ quality: 85 })
                .toBuffer();
            console.log(`✅ Resized Image ${id} from ${originalWidth}px to 1000px`);
        }

        // Upload 1000px version to Firebase
        const compressedPath = `compressed_${id}`;
        await bucket.file(compressedPath).save(compressedBuffer, {
            contentType: 'image/jpeg',
            metadata: {
                cacheControl: "public, max-age=31536000, immutable"
            },
        });

        // Save locally
        const localCompressedPath = path.join(__dirname, "warm-images", `${group_id}`, `compressed_${id}.jpg`);
        await fs.writeFile(localCompressedPath, compressedBuffer);

        // Handle elite plan - stripped version (full resolution, no metadata)
        let downloadURLStripped;
        if (planType === 'elite') {
            const strippedPath = `stripped_${id}`;
            await bucket.file(strippedPath).save(baseProcessedBuffer, {
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
            downloadURLStripped = downloadURLStripped_url;
        }

        // Handle 3000px version for non-lite plans
        let downloadURLCompressed_3k;
        if (planType !== 'lite') {
            if (originalWidth <= 3000) {
                // Use processed buffer as 3k version since it's already <= 3000px
                compressedBuffer3k = baseProcessedBuffer;
                console.log(`✅ Image ${id} is ${originalWidth}px wide (≤ 3000px), using original size`);
            } else {
                // Resize to 3000px
                compressedBuffer3k = await baseImage
                    .resize({ width: 3000 })
                    .jpeg({ quality: 90 })
                    .toBuffer();
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
            downloadURLCompressed_3k = downloadURLCompressed_3k_url;
        }

        // Create 200px thumbnail
        const thumbBuffer = await baseImage
            .resize({ width: 200 })
            .jpeg({ quality: 80 })
            .toBuffer();

        const thumbPath = `thumbnail_${id}`;
        await bucket.file(thumbPath).save(thumbBuffer, {
            contentType: 'image/jpeg',
            metadata: {
                cacheControl: "public, max-age=31536000, immutable"
            },
        });
        console.log(`✅ Created 200px thumbnail for Image ${id}`);

        // Get signed URLs
        const thumbFile = bucket.file(thumbPath);
        const compressedFile = bucket.file(compressedPath);

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
                signedUrlStripped: downloadURLStripped
            }
        };
    } catch (error) {
        console.error(`❌ Error processing Image ${id}:`, error.message);
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
            }
        };
    }
}

// Alternative approach: Add format validation at the beginning
async function validateImageFormat(buffer) {
    try {
        const metadata = await sharp(buffer).metadata();
        const supportedFormats = ['jpeg', 'jpg', 'png', 'webp', 'tiff', 'gif', 'svg'];

        if (!supportedFormats.includes(metadata.format)) {
            throw new Error(`Unsupported image format: ${metadata.format}`);
        }

        return {
            valid: true,
            format: metadata.format,
            width: metadata.width,
            height: metadata.height
        };
    } catch (error) {
        return {
            valid: false,
            error: error.message
        };
    }
}

// Enhanced version with format validation
async function processSingleImageWithValidation(image, planType) {
    const { id, firebase_path, filename, group_id, created_by_user, uploaded_at } = image;

    console.log(`🔃 Processing Image ${id} for group ${group_id}`);

    try {
        // Download image from Firebase
        const [originalBuffer] = await bucket.file(firebase_path).download();
        console.log(`✅ Downloaded Image ${id} from Firebase`);

        // Validate image format
        const formatValidation = await validateImageFormat(originalBuffer);
        if (!formatValidation.valid) {
            throw new Error(`Invalid image format: ${formatValidation.error}`);
        }

        console.log(`✅ Image ${id} validated - Format: ${formatValidation.format}, Size: ${formatValidation.width}x${formatValidation.height}`);

        // Continue with the rest of the processing...
        // (use the enhanced processSingleImage logic above)

    } catch (error) {
        console.error(`❌ Error processing Image ${id}:`, error.message);
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
            }
        };
    }
}