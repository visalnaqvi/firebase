// dbListener.js
const { Client } = require("pg");
const { processImages } = require("./S2_F1_sharp_standalone");

async function startListener() {
    const client = new Client({
        host: "ballast.proxy.rlwy.net",
        port: "56193",
        dbname: "railway",
        user: "postgres",
        password: "AfldldzckDWtkskkAMEhMaDXnMqknaPY"
    });

    await client.connect();

    // Listen to Postgres channel
    await client.query("LISTEN group_status_channel");

    console.log("Listening for group status events...");

    client.on("notification", async (msg) => {
        try {
            const payload = JSON.parse(msg.payload);
            console.log("Received event:", payload);

            if (payload.status === "heating") {
                await processImages(payload.group_id);
            }
        } catch (err) {
            console.error("Error handling event:", err);
        }
    });
}

startListener().catch(console.error);
