import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { exiftool } from "exiftool-vendored";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

async function processImage() {
    const inputPath = path.join(__dirname, "yo.jpg");       // original with metadata
    const strippedPath = path.join(__dirname, "output.jpg"); // stripped image
    const fileName = path.basename(inputPath, path.extname(inputPath));

    // Provided metadata (your JSON) - usually you would parse this from file/db
    const providedMeta = {
        ISO: 1250,
        Make: "SONY",
        Model: "ILCE-7M4",
        Artist: "Manav chahal",
        LensModel: "24-70mm F2.8 DG DN | Art 019",
        ExposureTime: 0.00625,
        FNumber: 7.1,
        FocalLength: 33.4,
        DateTimeOriginal: "2024:11:08 14:35:58",
        CreateDate: "2024:11:08 14:35:58",
        ModifyDate: "2024:11:08 14:35:58",
        Flash: "Flash fired, compulsory flash mode, return light not detected",
        Contrast: "Low",
        LensInfo: [
            24,
            70,
            2.8,
            2.8
        ],
        Software: "ILCE-7M4 v1.11",
        LensModel: "24-70mm F2.8 DG DN | Art 019",
        SceneType: "Directly photographed",
        Sharpness: "Low",
    };

    // Copy stripped image before attaching metadata
    const outputWithMeta = path.join(__dirname, `${fileName}_res.jpg`);
    fs.copyFileSync(strippedPath, outputWithMeta);

    // Write only supported tags
    await exiftool.write(outputWithMeta, providedMeta);

    console.log("✅ Metadata attached to:", outputWithMeta);

    await exiftool.end();
}

processImage().catch(console.error);
