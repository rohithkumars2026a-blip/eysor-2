import express from 'express';
import cors from 'cors';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import fs from 'fs/promises';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const app = express();
const PORT = process.env.PORT || 5000;

// Middleware
app.use(cors({
    origin: '*', // Allow all origins for worldwide access
    methods: ['GET', 'POST', 'PUT', 'DELETE'],
    allowedHeaders: ['Content-Type', 'Authorization']
}));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Database directory
const DB_DIR = join(__dirname, 'database');

// Initialize database directory
async function initDatabase() {
    try {
        await fs.mkdir(DB_DIR, { recursive: true });
        await fs.mkdir(join(DB_DIR, 'training_sessions'), { recursive: true });
        await fs.mkdir(join(DB_DIR, 'user_profiles'), { recursive: true });
        await fs.mkdir(join(DB_DIR, 'analytics'), { recursive: true });
        console.log('✅ Database directories initialized');
    } catch (error) {
        console.error('Database initialization error:', error);
    }
}

// ==================== TRAINING DATA ENDPOINTS ====================

// Save training session data
app.post('/api/training/save', async (req, res) => {
    try {
        const { userId, sessionId, calibrationData, modelMetrics, timestamp } = req.body;

        if (!userId || !sessionId || !calibrationData) {
            return res.status(400).json({ error: 'Missing required fields' });
        }

        const sessionData = {
            userId,
            sessionId,
            timestamp: timestamp || new Date().toISOString(),
            calibrationData,
            modelMetrics,
            samples: calibrationData.length,
            mode: req.body.mode || 'standard'
        };

        const filePath = join(DB_DIR, 'training_sessions', `${sessionId}.json`);
        await fs.writeFile(filePath, JSON.stringify(sessionData, null, 2));

        console.log(`💾 Saved training session: ${sessionId}`);
        res.json({ success: true, sessionId, message: 'Training data saved successfully' });
    } catch (error) {
        console.error('Error saving training data:', error);
        res.status(500).json({ error: 'Failed to save training data' });
    }
});

// Get training session by ID
app.get('/api/training/:sessionId', async (req, res) => {
    try {
        const { sessionId } = req.params;
        const filePath = join(DB_DIR, 'training_sessions', `${sessionId}.json`);

        const data = await fs.readFile(filePath, 'utf-8');
        res.json(JSON.parse(data));
    } catch (error) {
        if (error.code === 'ENOENT') {
            res.status(404).json({ error: 'Training session not found' });
        } else {
            res.status(500).json({ error: 'Failed to retrieve training data' });
        }
    }
});

// Get all training sessions for a user
app.get('/api/training/user/:userId', async (req, res) => {
    try {
        const { userId } = req.params;
        const files = await fs.readdir(join(DB_DIR, 'training_sessions'));

        const sessions = [];
        for (const file of files) {
            const data = await fs.readFile(join(DB_DIR, 'training_sessions', file), 'utf-8');
            const session = JSON.parse(data);
            if (session.userId === userId) {
                sessions.push(session);
            }
        }

        res.json({ userId, sessions, count: sessions.length });
    } catch (error) {
        res.status(500).json({ error: 'Failed to retrieve user sessions' });
    }
});

// ==================== USER PROFILE ENDPOINTS ====================

// Save or update user profile
app.post('/api/profile/save', async (req, res) => {
    try {
        const { userId, wearsGlasses, screenResolution, deviceInfo, preferences } = req.body;

        if (!userId) {
            return res.status(400).json({ error: 'userId is required' });
        }

        const profile = {
            userId,
            wearsGlasses: wearsGlasses || false,
            screenResolution,
            deviceInfo,
            preferences,
            createdAt: new Date().toISOString(),
            lastUpdated: new Date().toISOString()
        };

        const filePath = join(DB_DIR, 'user_profiles', `${userId}.json`);
        await fs.writeFile(filePath, JSON.stringify(profile, null, 2));

        res.json({ success: true, userId, message: 'Profile saved successfully' });
    } catch (error) {
        res.status(500).json({ error: 'Failed to save profile' });
    }
});

// Get user profile
app.get('/api/profile/:userId', async (req, res) => {
    try {
        const { userId } = req.params;
        const filePath = join(DB_DIR, 'user_profiles', `${userId}.json`);

        const data = await fs.readFile(filePath, 'utf-8');
        res.json(JSON.parse(data));
    } catch (error) {
        if (error.code === 'ENOENT') {
            res.status(404).json({ error: 'Profile not found' });
        } else {
            res.status(500).json({ error: 'Failed to retrieve profile' });
        }
    }
});

// ==================== ANALYTICS ENDPOINTS ====================

// Save analytics data
app.post('/api/analytics/save', async (req, res) => {
    try {
        const { userId, sessionId, metrics, timestamp } = req.body;

        const analyticsData = {
            userId,
            sessionId,
            timestamp: timestamp || new Date().toISOString(),
            metrics
        };

        const date = new Date().toISOString().split('T')[0];
        const filePath = join(DB_DIR, 'analytics', `${date}_${sessionId}.json`);
        await fs.writeFile(filePath, JSON.stringify(analyticsData, null, 2));

        res.json({ success: true, message: 'Analytics saved successfully' });
    } catch (error) {
        res.status(500).json({ error: 'Failed to save analytics' });
    }
});

// Get global statistics
app.get('/api/analytics/global', async (req, res) => {
    try {
        const trainingFiles = await fs.readdir(join(DB_DIR, 'training_sessions'));
        const profileFiles = await fs.readdir(join(DB_DIR, 'user_profiles'));
        const analyticsFiles = await fs.readdir(join(DB_DIR, 'analytics'));

        const stats = {
            totalTrainingSessions: trainingFiles.length,
            totalUsers: profileFiles.length,
            totalAnalyticsRecords: analyticsFiles.length,
            lastUpdated: new Date().toISOString()
        };

        res.json(stats);
    } catch (error) {
        res.status(500).json({ error: 'Failed to retrieve global statistics' });
    }
});

// Get aggregated metrics
app.get('/api/analytics/aggregated', async (req, res) => {
    try {
        const files = await fs.readdir(join(DB_DIR, 'training_sessions'));

        let totalSamples = 0;
        let standardModeCount = 0;
        let glassesModeCount = 0;

        for (const file of files) {
            const data = await fs.readFile(join(DB_DIR, 'training_sessions', file), 'utf-8');
            const session = JSON.parse(data);
            totalSamples += session.samples || 0;
            if (session.mode === 'standard') standardModeCount++;
            else if (session.mode === 'glasses') glassesModeCount++;
        }

        res.json({
            totalSessions: files.length,
            totalSamples,
            standardMode: standardModeCount,
            glassesMode: glassesModeCount,
            avgSamplesPerSession: files.length > 0 ? Math.round(totalSamples / files.length) : 0
        });
    } catch (error) {
        res.status(500).json({ error: 'Failed to retrieve aggregated metrics' });
    }
});

// ==================== PRE-TRAINED MODEL SHARING ====================

// Save pre-trained model weights (compressed)
app.post('/api/models/save', async (req, res) => {
    try {
        const { modelId, mode, weights, metadata } = req.body;

        if (!modelId || !mode || !weights) {
            return res.status(400).json({ error: 'Missing required fields' });
        }

        const modelData = {
            modelId,
            mode,
            weights,
            metadata,
            uploadedAt: new Date().toISOString()
        };

        const modelsDir = join(DB_DIR, 'pretrained_models');
        await fs.mkdir(modelsDir, { recursive: true });

        const filePath = join(modelsDir, `${modelId}_${mode}.json`);
        await fs.writeFile(filePath, JSON.stringify(modelData, null, 2));

        res.json({ success: true, modelId, message: 'Model weights saved successfully' });
    } catch (error) {
        res.status(500).json({ error: 'Failed to save model weights' });
    }
});

// Get pre-trained model weights
app.get('/api/models/:modelId/:mode', async (req, res) => {
    try {
        const { modelId, mode } = req.params;
        const filePath = join(DB_DIR, 'pretrained_models', `${modelId}_${mode}.json`);

        const data = await fs.readFile(filePath, 'utf-8');
        res.json(JSON.parse(data));
    } catch (error) {
        if (error.code === 'ENOENT') {
            res.status(404).json({ error: 'Model not found' });
        } else {
            res.status(500).json({ error: 'Failed to retrieve model' });
        }
    }
});

// ==================== HEALTH CHECK ====================

app.get('/api/health', (req, res) => {
    res.json({
        status: 'online',
        service: 'EYSOR Global Database',
        timestamp: new Date().toISOString(),
        uptime: process.uptime()
    });
});

app.get('/', (req, res) => {
    res.json({
        service: 'EYSOR Global Database API',
        version: '1.0.0',
        endpoints: {
            training: '/api/training/*',
            profile: '/api/profile/*',
            analytics: '/api/analytics/*',
            models: '/api/models/*',
            health: '/api/health'
        }
    });
});

// Start server
async function start() {
    await initDatabase();
    app.listen(PORT, '0.0.0.0', () => {
        console.log(`
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   🌐 EYSOR Global Database Server                        ║
║                                                           ║
║   Status: ONLINE                                         ║
║   Port: ${PORT}                                            ║
║   Access: http://0.0.0.0:${PORT}                          ║
║                                                           ║
║   📡 Accessible from any network worldwide               ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
        `);
        console.log(`📊 API Endpoints:`);
        console.log(`   POST /api/training/save - Save training session`);
        console.log(`   GET  /api/training/:sessionId - Get training data`);
        console.log(`   GET  /api/training/user/:userId - Get user sessions`);
        console.log(`   POST /api/profile/save - Save user profile`);
        console.log(`   GET  /api/profile/:userId - Get user profile`);
        console.log(`   POST /api/analytics/save - Save analytics`);
        console.log(`   GET  /api/analytics/global - Global statistics`);
        console.log(`   GET  /api/analytics/aggregated - Aggregated metrics`);
        console.log(`   POST /api/models/save - Save model weights`);
        console.log(`   GET  /api/models/:modelId/:mode - Get model weights`);
        console.log(`\n✅ Server ready for worldwide access!`);
    });
}

start().catch(console.error);
