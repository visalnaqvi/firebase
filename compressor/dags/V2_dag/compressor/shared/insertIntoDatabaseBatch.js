const updateStatusHistory = require("./updateStatusHistory")
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
        INSERT INTO images 
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

module.exports = insertIntoDatabaseBatch