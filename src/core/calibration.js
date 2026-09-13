export class CalibrationEngine {
    constructor() {
        this.calibrationPoints = this.generateCalibrationGrid();
        this.samples = [];
        this.overlay = null;
        this.progressText = null;

        this.SETTLING_TIME = 400; // ms to discard after point appears
        this.HOLD_TIME = 2500; // ms to collect samples
        this.SAMPLE_RATE = 60; // samples per second

        // Eye rest break configuration
        this.POINTS_BEFORE_BREAK = 3; // Take break every 3 points
        this.BREAK_DURATION = 10000; // 10 seconds break
    }

    initElements() {
        this.overlay = document.getElementById('calibration-overlay');
        this.progressText = document.getElementById('calibration-progress');
    }

    generateCalibrationGrid() {
        // 13-point calibration: 3x3 outer grid + 4 inner quadrant points
        const margin = 0.1; // 10% margin from edges
        const points = [];

        // 3x3 outer grid
        for (let row = 0; row < 3; row++) {
            for (let col = 0; col < 3; col++) {
                points.push({
                    x: margin + col * (1 - 2 * margin) / 2,
                    y: margin + row * (1 - 2 * margin) / 2
                });
            }
        }

        // 4 inner quadrant points
        const innerMargin = 0.3;
        points.push(
            { x: innerMargin, y: innerMargin },
            { x: 1 - innerMargin, y: innerMargin },
            { x: innerMargin, y: 1 - innerMargin },
            { x: 1 - innerMargin, y: 1 - innerMargin }
        );

        return points;
    }

    async run(audioGuide = null) {
        this.initElements();
        this.samples = [];

        if (!this.overlay || !this.progressText) {
            throw new Error('Calibration UI elements not found');
        }

        this.overlay.classList.add('active');

        if (audioGuide) {
            audioGuide.startCalibration(this.calibrationPoints.length);
        }

        try {
            for (let i = 0; i < this.calibrationPoints.length; i++) {
                const point = this.calibrationPoints[i];
                this.progressText.textContent =
                    `Calibration Point ${i + 1} / ${this.calibrationPoints.length}\nLook at the green dot`;

                if (audioGuide) {
                    audioGuide.nextPoint(i + 1, this.calibrationPoints.length);
                }

                const screenX = point.x * window.innerWidth;
                const screenY = point.y * window.innerHeight;

                await this.collectSamples(screenX, screenY, point);

                // Eye rest break after every POINTS_BEFORE_BREAK points
                if ((i + 1) % this.POINTS_BEFORE_BREAK === 0 && i + 1 < this.calibrationPoints.length) {
                    await this.takeEyeRestBreak(audioGuide);
                }
            }

            this.overlay.classList.remove('active');

            if (audioGuide) {
                audioGuide.calibrationComplete();
            }

            return this.samples;

        } catch (error) {
            this.overlay.classList.remove('active');
            throw error;
        }
    }

    async takeEyeRestBreak(audioGuide = null) {
        const breakSeconds = this.BREAK_DURATION / 1000;

        this.progressText.innerHTML = `
            <div style="font-size: 28px; margin-bottom: 20px;">😌 Eye Rest Break</div>
            <div style="font-size: 18px;">Close your eyes and relax for ${breakSeconds} seconds</div>
            <div id="break-countdown" style="font-size: 48px; margin-top: 20px; color: #4ade80;">${breakSeconds}</div>
        `;

        if (audioGuide) {
            audioGuide.eyeRestBreak(breakSeconds);
        }

        // Countdown timer
        const startTime = Date.now();
        const countdownInterval = setInterval(() => {
            const elapsed = Date.now() - startTime;
            const remaining = Math.ceil((this.BREAK_DURATION - elapsed) / 1000);

            const countdownEl = document.getElementById('break-countdown');
            if (countdownEl) {
                countdownEl.textContent = remaining;
            }

            if (remaining <= 0) {
                clearInterval(countdownInterval);
            }
        }, 100);

        await this.delay(this.BREAK_DURATION);

        if (audioGuide) {
            audioGuide.resumeCalibration();
        }

        this.progressText.innerHTML = `
            <div style="font-size: 24px;">Ready to continue!</div>
            <div style="font-size: 18px;">Look at the next green dot</div>
        `;

        await this.delay(1500); // Brief pause before next point
    }

    async collectSamples(screenX, screenY, normalizedPoint) {
        // Create calibration point with oscillating animation
        const pointElement = document.createElement('div');
        pointElement.className = 'calibration-point';
        pointElement.style.left = `${screenX}px`;
        pointElement.style.top = `${screenY}px`;
        this.overlay.appendChild(pointElement);

        // Wait for settling time
        await this.delay(this.SETTLING_TIME);

        // Collect samples
        const startTime = performance.now();
        const sampleInterval = 1000 / this.SAMPLE_RATE;
        const pointSamples = [];

        while (performance.now() - startTime < this.HOLD_TIME) {
            // Get current gaze data from vision engine (accessed via global reference)
            const gazeData = await this.getCurrentGazeData();

            if (gazeData) {
                pointSamples.push({
                    targetX: screenX,
                    targetY: screenY,
                    normalizedX: normalizedPoint.x,
                    normalizedY: normalizedPoint.y,
                    rawGazeX: gazeData.x,
                    rawGazeY: gazeData.y,
                    features: gazeData.features,
                    timestamp: performance.now()
                });
            }

            await this.delay(sampleInterval);
        }

        // Remove calibration point
        this.overlay.removeChild(pointElement);

        // Store all samples for this point
        this.samples.push(...pointSamples);

        // Brief pause before next point
        await this.delay(300);
    }

    async getCurrentGazeData() {
        // This will be called from app context where vision engine is available
        return new Promise((resolve) => {
            if (window.visionEngine && window.visionEngine.getLatestGaze) {
                resolve(window.visionEngine.getLatestGaze());
            } else {
                resolve(null);
            }
        });
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    getCalibrationData() {
        return this.samples;
    }
}
