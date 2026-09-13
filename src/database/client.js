/**
 * Global Database Client for EYSOR
 * Connects to worldwide database server for training data storage and retrieval
 */
export class GlobalDatabaseClient {
    constructor(serverUrl = 'http://localhost:5000') {
        this.serverUrl = serverUrl;
        this.userId = this.generateUserId();
    }

    generateUserId() {
        // Generate or retrieve from localStorage
        let userId = localStorage.getItem('eysor_user_id');
        if (!userId) {
            userId = 'user_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('eysor_user_id', userId);
        }
        return userId;
    }

    async saveTrainingSession(calibrationData, modelMetrics, mode) {
        const sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);

        const payload = {
            userId: this.userId,
            sessionId: sessionId,
            calibrationData: calibrationData,
            modelMetrics: modelMetrics,
            mode: mode,
            timestamp: new Date().toISOString(),
            screenResolution: {
                width: window.innerWidth,
                height: window.innerHeight
            },
            userAgent: navigator.userAgent
        };

        try {
            const response = await fetch(`${this.serverUrl}/api/training/save`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const result = await response.json();
            console.log('✅ Training session saved to global database:', sessionId);
            return result;
        } catch (error) {
            console.error('❌ Failed to save to global database:', error);
            // Fallback to local storage
            this.saveToLocalStorage(sessionId, payload);
            return { success: false, error: error.message, sessionId };
        }
    }

    async getUserSessions() {
        try {
            const response = await fetch(`${this.serverUrl}/api/training/user/${this.userId}`);
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Failed to fetch user sessions:', error);
            return { userId: this.userId, sessions: [], count: 0 };
        }
    }

    async saveUserProfile(wearsGlasses, preferences) {
        const profile = {
            userId: this.userId,
            wearsGlasses: wearsGlasses,
            screenResolution: {
                width: window.innerWidth,
                height: window.innerHeight,
                dpi: window.devicePixelRatio
            },
            deviceInfo: {
                userAgent: navigator.userAgent,
                platform: navigator.platform,
                language: navigator.language
            },
            preferences: preferences
        };

        try {
            const response = await fetch(`${this.serverUrl}/api/profile/save`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(profile)
            });

            const result = await response.json();
            console.log('✅ User profile saved to global database');
            return result;
        } catch (error) {
            console.error('Failed to save profile:', error);
            return { success: false, error: error.message };
        }
    }

    async getUserProfile() {
        try {
            const response = await fetch(`${this.serverUrl}/api/profile/${this.userId}`);
            const profile = await response.json();
            return profile;
        } catch (error) {
            console.error('Failed to fetch profile:', error);
            return null;
        }
    }

    async saveAnalytics(metrics) {
        const payload = {
            userId: this.userId,
            sessionId: 'analytics_' + Date.now(),
            metrics: metrics,
            timestamp: new Date().toISOString()
        };

        try {
            const response = await fetch(`${this.serverUrl}/api/analytics/save`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            return await response.json();
        } catch (error) {
            console.error('Failed to save analytics:', error);
            return { success: false };
        }
    }

    async getGlobalStatistics() {
        try {
            const response = await fetch(`${this.serverUrl}/api/analytics/global`);
            const stats = await response.json();
            return stats;
        } catch (error) {
            console.error('Failed to fetch global statistics:', error);
            return null;
        }
    }

    async getAggregatedMetrics() {
        try {
            const response = await fetch(`${this.serverUrl}/api/analytics/aggregated`);
            const metrics = await response.json();
            return metrics;
        } catch (error) {
            console.error('Failed to fetch aggregated metrics:', error);
            return null;
        }
    }

    async checkServerHealth() {
        try {
            const response = await fetch(`${this.serverUrl}/api/health`);
            const health = await response.json();
            return health;
        } catch (error) {
            console.error('Server health check failed:', error);
            return { status: 'offline' };
        }
    }

    saveToLocalStorage(sessionId, payload) {
        const key = `eysor_backup_${sessionId}`;
        try {
            localStorage.setItem(key, JSON.stringify(payload));
            console.log('💾 Saved to local storage as backup');
        } catch (error) {
            console.error('Local storage backup failed:', error);
        }
    }

    getUserId() {
        return this.userId;
    }
}
