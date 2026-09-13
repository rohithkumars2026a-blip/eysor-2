/**
 * OneEuro Filter Implementation
 * Dynamic low-pass filter that adapts to signal velocity
 * Eliminates jitter during fixations while maintaining responsiveness during saccades
 */
export class OneEuroFilter {
    constructor(minCutoff = 1.0, beta = 0.007, dCutoff = 1.0) {
        this.minCutoff = minCutoff;
        this.beta = beta;
        this.dCutoff = dCutoff;

        this.x = null;
        this.dx = null;
        this.lastTime = null;
    }

    filter(value, timestamp = performance.now()) {
        if (this.x === null) {
            this.x = value;
            this.dx = 0;
            this.lastTime = timestamp;
            return value;
        }

        const dt = (timestamp - this.lastTime) / 1000; // Convert to seconds
        if (dt <= 0) return this.x;

        // Calculate velocity (derivative)
        const edValue = (value - this.x) / dt;
        const edAlpha = this.smoothingFactor(dt, this.dCutoff);
        this.dx = this.exponentialSmoothing(edAlpha, edValue, this.dx);

        // Calculate adaptive cutoff frequency
        const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
        const alpha = this.smoothingFactor(dt, cutoff);

        // Apply filter
        this.x = this.exponentialSmoothing(alpha, value, this.x);
        this.lastTime = timestamp;

        return this.x;
    }

    smoothingFactor(dt, cutoff) {
        const r = 2 * Math.PI * cutoff * dt;
        return r / (r + 1);
    }

    exponentialSmoothing(alpha, x, xPrev) {
        return alpha * x + (1 - alpha) * xPrev;
    }

    reset() {
        this.x = null;
        this.dx = null;
        this.lastTime = null;
    }
}

/**
 * Dual-axis filter for (x, y) coordinates
 */
export class GazeFilter {
    constructor(minCutoff = 1.0, beta = 0.007, dCutoff = 1.0) {
        this.filterX = new OneEuroFilter(minCutoff, beta, dCutoff);
        this.filterY = new OneEuroFilter(minCutoff, beta, dCutoff);
    }

    filter(x, y, timestamp = performance.now()) {
        return {
            x: this.filterX.filter(x, timestamp),
            y: this.filterY.filter(y, timestamp)
        };
    }

    reset() {
        this.filterX.reset();
        this.filterY.reset();
    }
}
