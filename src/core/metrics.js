export class MetricsEngine {
    constructor() {
        this.validationPoints = [];
        this.results = [];

        // Screen DPI (approximate, can be refined)
        this.dpi = this.calculateDPI();
        this.viewingDistance = 600; // mm (typical viewing distance)
    }

    calculateDPI() {
        // Create a 1-inch div and measure its pixel size
        const div = document.createElement('div');
        div.style.width = '1in';
        div.style.visibility = 'hidden';
        document.body.appendChild(div);
        const dpi = div.offsetWidth;
        document.body.removeChild(div);
        return dpi || 96; // Default to 96 if measurement fails
    }

    generateValidationPoints(count = 5) {
        const points = [];
        const margin = 0.15;

        // Generate random points across screen
        for (let i = 0; i < count; i++) {
            points.push({
                x: (margin + Math.random() * (1 - 2 * margin)) * window.innerWidth,
                y: (margin + Math.random() * (1 - 2 * margin)) * window.innerHeight
            });
        }

        return points;
    }

    async verify(gazeProvider) {
        this.validationPoints = this.generateValidationPoints(5);
        this.results = [];

        const overlay = document.getElementById('calibration-overlay');
        const progressText = document.getElementById('calibration-progress');
        overlay.classList.add('active');

        try {
            for (let i = 0; i < this.validationPoints.length; i++) {
                const point = this.validationPoints[i];
                progressText.textContent =
                    `Verification ${i + 1} / ${this.validationPoints.length}\nLook at the green dot`;

                const error = await this.measurePoint(point, gazeProvider);
                this.results.push(error);
            }

            overlay.classList.remove('active');
            return this.calculateMetrics();

        } catch (error) {
            overlay.classList.remove('active');
            throw error;
        }
    }

    async measurePoint(point, gazeProvider) {
        // Display validation point
        const pointElement = document.createElement('div');
        pointElement.className = 'calibration-point';
        pointElement.style.left = `${point.x}px`;
        pointElement.style.top = `${point.y}px`;
        document.getElementById('calibration-overlay').appendChild(pointElement);

        // Wait for user to fixate
        await this.delay(500);

        // Collect samples
        const samples = [];
        const sampleCount = 30;

        for (let i = 0; i < sampleCount; i++) {
            const gaze = await gazeProvider(point);
            if (gaze) {
                samples.push({
                    x: gaze.x,
                    y: gaze.y
                });
            }
            await this.delay(16); // ~60 FPS
        }

        // Remove point
        pointElement.remove();
        await this.delay(200);

        // Calculate average gaze position
        const avgGaze = {
            x: samples.reduce((sum, s) => sum + s.x, 0) / samples.length,
            y: samples.reduce((sum, s) => sum + s.y, 0) / samples.length
        };

        // Calculate error
        const error = this.euclideanDistance(point, avgGaze);

        return {
            target: point,
            measured: avgGaze,
            error: error,
            samples: samples
        };
    }

    calculateMetrics() {
        // Mean Euclidean Error
        const errors = this.results.map(r => r.error);
        const meanError = errors.reduce((sum, e) => sum + e, 0) / errors.length;

        // Standard Deviation (precision)
        const variance = errors.reduce((sum, e) => sum + Math.pow(e - meanError, 2), 0) / errors.length;
        const stdDev = Math.sqrt(variance);

        // Convert to millimeters
        const meanErrorMM = (meanError / this.dpi) * 25.4;

        // Calculate Visual Angle Error (degrees)
        const visualAngle = this.pixelsToVisualAngle(meanError);

        return {
            meanError: meanError,
            meanErrorMM: meanErrorMM,
            stdDev: stdDev,
            visualAngle: visualAngle,
            rawResults: this.results
        };
    }

    euclideanDistance(p1, p2) {
        return Math.sqrt(Math.pow(p2.x - p1.x, 2) + Math.pow(p2.y - p1.y, 2));
    }

    pixelsToVisualAngle(pixels) {
        // Convert pixels to mm
        const mm = (pixels / this.dpi) * 25.4;

        // Calculate visual angle using tan(θ) = opposite / adjacent
        const angleRadians = Math.atan(mm / this.viewingDistance);
        const angleDegrees = (angleRadians * 180) / Math.PI;

        return angleDegrees;
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}
