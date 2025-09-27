// CommonJS
const { Pool } = require("pg");
const { spawn } = require("child_process");
const { join } = require("path");
const updateStatusHistory = require("./shared/updateStatusHistory");
const updateProcessStatus = require("./shared/updateProcessStatus");

const serviceAccount = require("../firebase-key.json");
const { fileURLToPath } = require("url");
const { basename } = require("path");
const admin = require("firebase-admin");

const pool = new Pool({
    // Your database connection
    connectionString: "postgresql://postgres:AfldldzckDWtkskkAMEhMaDXnMqknaPY@ballast.proxy.rlwy.net:56193/railway"
    // connectionString: "postgresql://postgres:kdVrNTrtLzzAaOXzKHaJCzhmoHnSDKDG@nozomi.proxy.rlwy.net:24794/railway"
});
admin.initializeApp({
    credential: admin.credential.cert(serviceAccount),
    // DEV BUCKET
    storageBucket: 'gallery-585ee.firebasestorage.app',
    // PROD BUCKET
    // storageBucket: 'gallery-585ee-production',
});
const bucket = admin.storage().bucket();
class ProcessingOrchestrator {
    constructor() {
        this.isRunning = false;
        this.currentProcess = null;
    }

    async runScript(scriptPath, scriptName) {
        return new Promise((resolve, reject) => {
            console.log(`🚀 Starting ${scriptName} processing...`);

            const child = spawn('node', [scriptPath], {
                stdio: 'pipe',
                cwd: process.cwd()
            });

            let stdout = '';
            let stderr = '';

            child.stdout.on('data', (data) => {
                const output = data.toString();
                stdout += output;
                console.log(`[${scriptName}] ${output.trim()}`);
            });

            child.stderr.on('data', (data) => {
                const output = data.toString();
                stderr += output;
                console.error(`[${scriptName}] ERROR: ${output.trim()}`);
            });

            child.on('close', (code) => {
                if (code === 0) {
                    console.log(`✅ ${scriptName} processing completed successfully`);
                    resolve({ success: true, stdout, stderr });
                } else {
                    console.error(`❌ ${scriptName} processing failed with code ${code}`);
                    resolve({ success: false, code, stdout, stderr });
                }
            });

            child.on('error', (error) => {
                console.error(`❌ Failed to start ${scriptName}:`, error);
                reject(error);
            });

            this.currentProcess = child;
        });
    }

    async checkForWork(client) {
        try {
            // Check for Firebase images
            const [firebaseFiles] = await bucket.getFiles({
                prefix: 'u_',
                maxResults: 1
            });

            // Check for unprocessed Drive folders
            const { rows: driveFolders } = await client.query(
                'SELECT COUNT(*) as count FROM drive_folders WHERE is_processed = false LIMIT 1'
            );

            return {
                hasFirebaseWork: firebaseFiles.length > 0,
                hasDriveWork: parseInt(driveFolders[0].count) > 0
            };
        } catch (error) {
            console.error('Error checking for work:', error);
            throw Error("Something went wront while running checkForWork in orch " + error.message);
        }
    }

    async processImages() {
        if (this.isRunning) {
            console.log('⏸️ Processing already running, skipping...');
            return;
        }
        const client = await pool.connect();
        this.isRunning = true;
        const runId = Date.now();

        try {
            await updateStatusHistory(client, runId, 'orchestrator', 'cycle_start', null, null, null, null, 'Starting new processing cycle');
            await updateProcessStatus(client, "node_compression", "healthy", null, "running", false)
            const workCheck = await this.checkForWork(client);

            if (!workCheck.hasFirebaseWork && !workCheck.hasDriveWork) {
                console.log('⏸️ No work found for either Firebase or Drive, waiting...');
                await updateStatusHistory(client, runId, 'orchestrator', 'cycle_start', 0, 0, 0, null, 'no_group');
                await updateProcessStatus(client, 'firebase_compression', 'done', null, "", false);
                return;
            }

            let firebaseResult = { success: true };
            let driveResult = { success: true };


            // Process Drive images (if any and Firebase succeeded)
            if (workCheck.hasDriveWork) {
                if (!firebaseResult.success) {
                    console.log('⚠️ Skipping Drive processing due to Firebase failure');
                    await updateStatusHistory(client, runId, 'orchestrator', 'drive_skipped', null, null, null, null, 'Drive processing skipped due to Firebase failure');
                } else {
                    console.log('🔃 Starting Drive image processing...');
                    await updateProcessStatus(client, 'drive_compression', 'running');

                    driveResult = await this.runScript(
                        join(__dirname, 'drive.js'),
                        'Drive'
                    );

                    if (driveResult.success) {
                        await updateProcessStatus(client, 'drive_compression', 'completed');
                        await updateStatusHistory(client, runId, 'orchestrator', 'drive_complete', null, null, null, null, 'Drive processing completed');
                    } else {
                        await updateProcessStatus(client, 'drive_compression', 'failed', null, 'Drive script failed');
                        await updateStatusHistory(client, runId, 'orchestrator', 'drive_failed', null, null, null, null, `Drive processing failed: ${driveResult.stderr}`);
                    }
                }
            }

            // Process Firebase images first (if any)
            if (workCheck.hasFirebaseWork) {
                console.log('🔃 Starting Firebase image processing...');
                await updateProcessStatus(client, 'firebase_compression', 'running', null, "", false);

                firebaseResult = await this.runScript(
                    join(__dirname, 'firebase.js'),
                    'Firebase'
                );

                if (firebaseResult.success) {
                    await updateProcessStatus('firebase_compression', 'completed');
                    await updateStatusHistory(client, runId, 'orchestrator', 'firebase_complete', null, null, null, null, 'Firebase processing completed');
                } else {
                    await updateProcessStatus('firebase_compression', 'failed', null, 'Firebase script failed');
                    await updateStatusHistory(client, runId, 'orchestrator', 'firebase_failed', null, null, null, null, `Firebase processing failed: ${firebaseResult.stderr}`);
                }
            }
            console.log(`🎉 Processing cycle completed - Firebase: ${firebaseResult.success ? '✅' : '❌'}, Drive: ${driveResult.success ? '✅' : '❌'}`);

        } catch (error) {
            console.error('❌ Orchestrator error:', error);
            await updateStatusHistory(client, runId, 'orchestrator', 'error', null, null, null, null, `Orchestrator error: ${error.message}`);
            await updateProcessStatus(client, 'orchestrator', 'failed', null, error.message, true);
        } finally {
            this.isRunning = false;
            this.currentProcess = null;
        }
    }

    async start() {
        console.log('🎬 Starting Image Processing Orchestrator...');

        while (true) {
            try {
                await this.processImages();

                // Wait 5 minutes before next cycle
                console.log('⏸️ Waiting 5 minutes before next cycle...');
                await new Promise(resolve => setTimeout(resolve, 5 * 60 * 1000));

            } catch (error) {
                console.error('❌ Unexpected orchestrator error:', error);
                console.log('⏸️ Waiting 30 seconds before retry...');
                await new Promise(resolve => setTimeout(resolve, 30 * 1000));
            }
        }
    }

    // Graceful shutdown
    async shutdown() {
        console.log('🛑 Shutting down orchestrator...');
        if (this.currentProcess) {
            this.currentProcess.kill('SIGTERM');
        }
        await pool.end();
        process.exit(0);
    }
}

// Handle shutdown signals
const orchestrator = new ProcessingOrchestrator();

// Start if this file is run directly

if (require.main === module) {
    orchestrator.start().catch(console.error);
}

module.exports = ProcessingOrchestrator;