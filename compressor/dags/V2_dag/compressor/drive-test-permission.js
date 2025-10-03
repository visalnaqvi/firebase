const { google } = require("googleapis");

const SCOPES = ["https://www.googleapis.com/auth/drive.readonly"];
const auth = new google.auth.GoogleAuth({
    keyFile: "./snapper-gallery-75b9ac194113.json",
    scopes: SCOPES,
});

const drive = google.drive({ version: "v3", auth });

// List all folders shared with me by a specific email
async function listFoldersSharedBy(email) {
    let folders = [];
    let pageToken = null;

    do {
        const res = await drive.files.list({
            q: "mimeType='application/vnd.google-apps.folder' and sharedWithMe",
            fields: "nextPageToken, files(id, name, owners, permissions)",
            pageSize: 100,
            pageToken,
            supportsAllDrives: true,
            includeItemsFromAllDrives: true,
        });
        console.log("res ", JSON.stringify(res.data))
        for (const f of res.data.files) {
            console.log("file ", f.name)
            // Check if the folder is owned/shared by the email
            const ownerMatch = f.owners?.some(o => o.emailAddress.toLowerCase() === email.toLowerCase());
            const permissionMatch = f.permissions?.some(p => p.emailAddress?.toLowerCase() === email.toLowerCase());

            if (ownerMatch || permissionMatch) {
                folders.push({ id: f.id, name: f.name });
            }
        }

        pageToken = res.data.nextPageToken;
    } while (pageToken);

    return folders;
}

// Example usage
const main = async () => {
    const sharedByEmail = "visal.shadi@gmail.com";
    console.log("ll")
    const folders = await listFoldersSharedBy(sharedByEmail);
    console.log("Folders shared by", sharedByEmail, ":", folders);
};
if (require.main === module) {
    main()
}

