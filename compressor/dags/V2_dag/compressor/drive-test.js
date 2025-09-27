const { google } = require("googleapis");
const fs = require("fs");
const path = require("path");

// Path to your service account JSON
const KEYFILEPATH = "./buttons-2dc4a-866e9e3e7b0a.json";

// Scopes required
const SCOPES = ["https://www.googleapis.com/auth/drive.readonly"];

// Authenticate with service account
const auth = new google.auth.GoogleAuth({
    keyFile: KEYFILEPATH,
    scopes: SCOPES,
});

const drive = google.drive({ version: "v3", auth });

// List all files inside a folder
async function listFilesInFolder(folderId) {
    const res = await drive.files.list({
        q: `'${folderId}' in parents and trashed = false`,
        fields: "files(id, name, mimeType, size)",
    });
    return res.data.files || [];
}

// Download file by ID
async function downloadFile(fileId, destPath) {
    const dest = fs.createWriteStream(destPath);
    const res = await drive.files.get(
        { fileId, alt: "media" },
        { responseType: "stream" }
    );

    return new Promise((resolve, reject) => {
        res.data
            .on("end", () => {
                console.log(`Downloaded ${destPath}`);
                resolve();
            })
            .on("error", (err) => reject(err))
            .pipe(dest);
    });
}

// Example usage
async function main() {
    try {
        const folderId = "1danu_-S6m5SXuiks0uTauKyCBx28AG4c"; // extract from user-provided link
        const files = await listFilesInFolder(folderId);

        console.log("Files in folder:", files);

        for (const f of files) {
            const des = path.join(
                __dirname,
                "..",
                "warm-images",
                "test",
                "compressed_" + f.id + ".jpg"
            );
            await downloadFile(f.id, des);
        }
    } catch (err) {
        console.error("Error:", err.message);
    }
}

// Run if file is called directly
if (require.main === module) {
    main();
}
