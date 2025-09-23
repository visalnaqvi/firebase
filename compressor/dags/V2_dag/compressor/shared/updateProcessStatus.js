async function updateProcessStatus(client, task, status, processingGroup = null, failReason = null, isIdeal = false) {
    try {
        await client.query(
            `UPDATE process_status 
                 SET task_status = $1, 
                     processing_group = $2, 
                     fail_reason = $3, 
                     ended_at = NOW(), 
                     is_ideal = $4
                 WHERE task = $5`,
            [status, processingGroup, failReason, isIdeal, task]
        );
    } catch (error) {
        console.error(`Error updating process status for ${task}:`, error);
    }
}

module.exports = updateProcessStatus