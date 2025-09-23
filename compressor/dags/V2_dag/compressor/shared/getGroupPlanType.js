async function getGroupPlanType(client, groupIds) {
    try {
        const { rows } = await client.query(
            `SELECT id, plan_type FROM groups WHERE id = ANY($1)`,
            [groupIds]
        );

        const planDetails = rows.map(r => ({
            groupId: r.id,
            planType: r.plan_type
        }));
        console.log("got group plan type successfullt: " + planDetails)
        return {
            success: true,
            planDetails: planDetails
        }
    } catch (error) {
        console.error("Error getting plan type:", error);
        return { success: false, errorReason: "getting plan type for groups", error: error.message };
    }
}

module.exports = getGroupPlanType