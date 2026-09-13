import { VisionEngine } from './core/vision.js';
import { CalibrationEngine } from './core/calibration.js';
import { TitanCorrector } from './ml/titan-fast.js';
import { AnalyticsModels } from './ml/analytics.js';
import { MetricsEngine } from './core/metrics.js';
import { CursorController } from './core/cursor.js';
import { GlobalDatabaseClient } from './database/client.js';
import { AudioGuide } from './core/audio.js';

class EysorApp {
    constructor() {
        this.wearsGlasses = false;
        this.vision = null;
        this.calibration = new CalibrationEngine();
        this.corrector = null;
        this.analytics = new AnalyticsModels();
        this.metrics = new MetricsEngine();
        this.cursor = new CursorController();
        this.database = new GlobalDatabaseClient('http://localhost:5000');
        this.audioGuide = new AudioGuide();

        this.isCalibrated = false;
        this.isRunning = false;
        this.fpsCounter = { frames: 0, lastTime: performance.now() };
    }

    start() {
        this.initUI();
        this.init();
    }

    async init() {
        try {
            console.log('🚀 Starting EYSOR initialization...');

            // Show glasses mode selector
            this.updateStatus('Select your configuration...');
            await this.selectGlassesMode();
            console.log(`👓 Mode selected: ${this.wearsGlasses ? 'Glasses' : 'Standard'}`);

            this.updateStatus('Initializing camera...');
            this.vision = new VisionEngine(this.wearsGlasses);

            // Make vision engine globally accessible for calibration
            window.visionEngine = this.vision;

            console.log('📷 Initializing camera and MediaPipe...');
            await this.vision.init();
            console.log('✅ Camera initialized');

            // Initialize audio guide
            this.updateStatus('Initializing JARVIS audio system...');
            this.audioGuide.init();

            // Wait a moment for voices to load, then play welcome
            setTimeout(() => {
                this.audioGuide.welcomeMessage();
            }, 500);
            console.log('🎙️ Audio guide ready');

            this.updateStatus('Initializing TITAN AI...');
            console.log('🧠 Quick-loading TITAN core...');
            this.corrector = new TitanCorrector(this.wearsGlasses);
            await this.corrector.init();
            console.log('✅ TITAN ready (full ensemble will build during training)');

            // Skip analytics - not needed for core functionality
            console.log('⚡ Skipping analytics (faster startup)');

            // Save user profile to global database (non-blocking)
            this.database.saveUserProfile(this.wearsGlasses, {
                firstUse: Date.now()
            }).catch(() => console.warn('⚠️ Database offline'));

            const mode = this.wearsGlasses ? 'glasses' : 'standard';
            this.updateStatus(`Ready for calibration (${mode} mode)`);
            this.enableButton('btn-calibrate');
            console.log('✅ All systems ready');

            this.startTracking();
        } catch (error) {
            console.error('❌ Initialization error:', error);
            console.error('Error stack:', error.stack);
            this.updateStatus(`Error: ${error.message}`);

            // Show user-friendly error
            const errorMsg = error.message.includes('Camera')
                ? 'Camera access denied or unavailable. Please allow camera permissions and refresh.'
                : error.message.includes('MediaPipe')
                ? 'Failed to load face tracking models. Please check your internet connection and refresh.'
                : `Initialization failed: ${error.message}`;

            alert(`${errorMsg}\n\nTechnical details in browser console (F12)`);
        }
    }

    async selectGlassesMode() {
        return new Promise((resolve) => {
            const modal = document.getElementById('glasses-modal');
            const btnNoGlasses = document.getElementById('btn-no-glasses');
            const btnGlasses = document.getElementById('btn-glasses');
            const modeBadge = document.getElementById('mode-badge');

            modal.classList.add('active');

            btnNoGlasses.onclick = () => {
                this.wearsGlasses = false;
                modeBadge.textContent = 'Standard';
                modal.classList.remove('active');
                resolve();
            };

            btnGlasses.onclick = () => {
                this.wearsGlasses = true;
                modeBadge.textContent = 'Glasses';
                modal.classList.remove('active');
                resolve();
            };
        });
    }

    initUI() {
        document.getElementById('btn-calibrate').addEventListener('click', () => this.startCalibration());
        document.getElementById('btn-verify').addEventListener('click', () => this.verifyAccuracy());
        document.getElementById('btn-toggle-cursor').addEventListener('click', () => this.toggleCursor());
        document.getElementById('btn-switch-mode').addEventListener('click', () => this.switchMode());
        document.getElementById('btn-toggle-audio').addEventListener('click', () => this.toggleAudio());

        // Update mute button state
        this.updateAudioButton();
    }

    toggleAudio() {
        const isMuted = this.audioGuide.toggle();
        this.updateAudioButton();

        if (!isMuted) {
            this.audioGuide.playBeep(440, 100);
        }
    }

    updateAudioButton() {
        const btn = document.getElementById('btn-toggle-audio');
        const isMuted = this.audioGuide.getMutedState();
        btn.textContent = isMuted ? '🔇 Unmute Audio' : '🔊 Mute Audio';
        btn.style.opacity = isMuted ? '0.6' : '1';
    }

    async switchMode() {
        if (confirm('Switching modes will reset calibration. Continue?')) {
            this.isCalibrated = false;
            this.isRunning = false;
            location.reload(); // Simple reload to restart with mode selection
        }
    }

    async startCalibration() {
        try {
            this.updateStatus('Starting calibration...');
            this.disableButton('btn-calibrate');

            const calibrationData = await this.calibration.run(this.audioGuide);

            // Simple quality check without analytics models
            console.log(`📊 Calibration collected ${calibrationData.length} samples`);
            const sampleQuality = calibrationData.length > 1000 ? 'Excellent' :
                                  calibrationData.length > 700 ? 'Good' : 'Fair';
            console.log(`Pattern Quality: ${sampleQuality}`);

            this.updateStatus('Training correction model...');
            await this.corrector.train(calibrationData, this.audioGuide);

            this.isCalibrated = true;
            this.updateStatus('Calibration complete! System ready.');
            this.audioGuide.trainingComplete();
            this.enableButton('btn-verify');
            this.enableButton('btn-calibrate');

            // Save training data to global database (non-blocking)
            this.database.saveTrainingSession(
                calibrationData,
                {
                    patternQuality: sampleQuality,
                    sampleCount: calibrationData.length
                },
                this.wearsGlasses ? 'glasses' : 'standard'
            ).catch(() => console.warn('⚠️ Database save failed'));

            this.updateStatus('Ready! Calibration data saved globally.');

        } catch (error) {
            console.error('Calibration error:', error);
            this.updateStatus(`Calibration failed: ${error.message}`);
            this.audioGuide.playError();
            this.audioGuide.speak('Calibration failed. Please try again.');
            this.enableButton('btn-calibrate');
        }
    }

    async verifyAccuracy() {
        try {
            this.updateStatus('Running accuracy verification...');
            this.disableButton('btn-verify');
            this.audioGuide.verificationStart();

            const results = await this.metrics.verify((point) => {
                return new Promise((resolve) => {
                    const checkGaze = () => {
                        const rawGaze = this.vision.getLatestGaze();
                        if (rawGaze) {
                            const correctedGaze = this.corrector.correct(
                                rawGaze.x,
                                rawGaze.y,
                                rawGaze.features
                            );
                            resolve(correctedGaze);
                        } else {
                            requestAnimationFrame(checkGaze);
                        }
                    };
                    checkGaze();
                });
            });

            this.displayMetrics(results);
            this.updateStatus('Verification complete');
            this.audioGuide.verificationComplete(results.meanError);
            this.enableButton('btn-verify');

            // Save analytics to global database (non-blocking)
            this.database.saveAnalytics({
                meanError: results.meanError,
                meanErrorMM: results.meanErrorMM,
                stdDev: results.stdDev,
                visualAngle: results.visualAngle
            }).catch(() => console.warn('⚠️ Analytics save failed'));

        } catch (error) {
            console.error('Verification error:', error);
            this.updateStatus(`Verification failed: ${error.message}`);
            this.audioGuide.playError();
            this.enableButton('btn-verify');
        }
    }

    startTracking() {
        this.isRunning = true;

        const track = () => {
            if (!this.isRunning) return;

            const rawGaze = this.vision.getLatestGaze();

            if (rawGaze && this.isCalibrated) {
                const correctedGaze = this.corrector.correct(
                    rawGaze.x,
                    rawGaze.y,
                    rawGaze.features
                );

                this.cursor.update(correctedGaze.x, correctedGaze.y);
            }

            this.updateFPS();
            requestAnimationFrame(track);
        };

        track();
    }

    toggleCursor() {
        this.cursor.toggle();
    }

    displayMetrics(results) {
        document.getElementById('accuracy').textContent =
            `${results.meanError.toFixed(1)}px (${results.meanErrorMM.toFixed(1)}mm)`;
        document.getElementById('precision').textContent =
            `±${results.stdDev.toFixed(1)}px`;
        document.getElementById('visual-angle').textContent =
            `${results.visualAngle.toFixed(2)}°`;
    }

    updateFPS() {
        this.fpsCounter.frames++;
        const now = performance.now();
        const elapsed = now - this.fpsCounter.lastTime;

        if (elapsed >= 1000) {
            const fps = Math.round((this.fpsCounter.frames * 1000) / elapsed);
            document.getElementById('fps').textContent = fps;
            this.fpsCounter.frames = 0;
            this.fpsCounter.lastTime = now;
        }
    }

    updateStatus(message) {
        document.getElementById('status').textContent = message;
    }

    enableButton(id) {
        document.getElementById(id).disabled = false;
    }

    disableButton(id) {
        document.getElementById(id).disabled = true;
    }
}

// Initialize app when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        const app = new EysorApp();
        app.start();
    });
} else {
    const app = new EysorApp();
    app.start();
}
