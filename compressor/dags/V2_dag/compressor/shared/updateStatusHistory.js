async function updateStatusHistory(client, run_id, task, sub_task, totalImagesInitialized, totalImagesFailed, totalImagesProcessed, groupId, fail_reason) {
    try {
        await client.query(
            `INSERT INTO process_history 
            (worker_id, run_id, task, sub_task, initialized_count, success_count, failed_count, group_id, ended_at, fail_reason)
            VALUES (1, $1, $2, $3, $4, $5, $6, $7, NOW(), $8)`,
            [run_id, task, sub_task, totalImagesInitialized, totalImagesProcessed, totalImagesFailed, groupId, fail_reason]
        );

        return { success: true };
    } catch (error) {
        console.error("Error inserting into process_history:", error);
        return { success: false, errorReason: "updating history", error: error.message };
    }
}

module.exports = updateStatusHistory; 