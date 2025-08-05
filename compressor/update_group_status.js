const { Client } = require('pg');
const dayjs = require('dayjs');

const client = new Client({
    connectionString: process.env.POSTGRES_URL,
});

(async () => {
    try {
        await client.connect();
        console.log('✅ Connected to PostgreSQL');

        // Step 1: Get all groups with 'heating' status
        const groupsRes = await client.query(`SELECT group_id FROM groups WHERE status = 'heating'`);
        const heatingGroups = groupsRes.rows.map(row => row.group_id);

        const heatedGroups = [];

        for (const groupId of heatingGroups) {
            // Step 2: Get latest image for group
            const imageRes = await client.query(
                `SELECT created_at FROM images WHERE group_id = $1 ORDER BY created_at DESC LIMIT 1`,
                [groupId]
            );

            if (imageRes.rows.length === 0) continue; // no images

            const latestCreatedAt = dayjs(imageRes.rows[0].created_at);
            const oneHourAgo = dayjs().subtract(1, 'hour');

            // Step 3: Check if older than 1 hour
            if (latestCreatedAt.isBefore(oneHourAgo)) {
                heatedGroups.push(groupId);
            }
        }

        console.log(`🔥 Found ${heatedGroups.length} heated groups`);

        // Step 4: For each heated group, get total images and total size
        for (const groupId of heatedGroups) {
            const statsRes = await client.query(
                `SELECT COUNT(*) AS total_images, COALESCE(SUM(size), 0) AS total_size FROM images WHERE group_id = $1`,
                [groupId]
            );

            const totalImages = parseInt(statsRes.rows[0].total_images);
            const totalSize = parseInt(statsRes.rows[0].total_size);

            // Step 5: Update group
            await client.query(
                `UPDATE groups SET status = 'hot', total_images = $1, total_size = $2 WHERE group_id = $3`,
                [totalImages, totalSize, groupId]
            );
        }

        console.log('✅ Done updating all heated groups');

    } catch (err) {
        console.error('❌ Error:', err);
    } finally {
        await client.end();
        console.log('🔌 Disconnected from PostgreSQL');
    }
})();
