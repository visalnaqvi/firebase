import { exiftool } from "exiftool-vendored";
import fetch from "node-fetch";
import fs from "fs/promises";

async function copyMetadata(firebaseUrl, targetPath) {
    try {
        // 1. Download image from Firebase Storage URL
        const response = await fetch(firebaseUrl);
        if (!response.ok) {
            throw new Error(`Failed to fetch image: ${response.statusText}`);
        }
        const buffer = Buffer.from(await response.arrayBuffer());

        // Save temporarily
        const tempFile = "temp_firebase.jpg";
        await fs.writeFile(tempFile, buffer);

        // 2. Read metadata from downloaded image
        const metadata = await exiftool.read(tempFile);
        console.log("✅ Extracted metadata:", metadata);

        // 3. Write metadata into another image (output.jpg)
        await exiftool.write(targetPath, metadata);

        console.log(`✅ Metadata copied into ${targetPath}`);

        // Cleanup
        await fs.unlink(tempFile);
        await exiftool.end();
    } catch (err) {
        console.error("❌ Error:", err);
    }
}

// Example usage
const firebaseUrl = "https://firebasestorage.googleapis.com/v0/b/gallery-585ee.firebasestorage.app/o/a8406cc4-5dd5-4a66-bab0-17b63cfdbcb2?alt=media&token=11db51bc-9c03-4138-a818-462315f15df4";
const targetPath = "output.jpg";

copyMetadata(firebaseUrl, targetPath);
