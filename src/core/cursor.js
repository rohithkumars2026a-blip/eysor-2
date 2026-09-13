export class CursorController {
    constructor() {
        this.cursorElement = document.getElementById('gaze-cursor');
        this.dwellCanvas = document.getElementById('dwell-timer');
        this.dwellCtx = this.dwellCanvas.getContext('2d');

        this.isVisible = true;
        this.currentX = 0;
        this.currentY = 0;

        // Dwell click settings
        this.isDwelling = false;
        this.dwellStartTime = null;
        this.dwellDuration = 1200; // ms
        this.dwellRadius = 25; // px
        this.dwellPositions = [];

        this.lastClickTime = 0;
        this.clickCooldown = 500; // ms between clicks
    }

    update(x, y) {
        this.currentX = x;
        this.currentY = y;

        // Update cursor position using GPU acceleration
        this.cursorElement.style.transform = `translate3d(${x}px, ${y}px, 0) translate(-50%, -50%)`;

        // Check for dwell click
        this.checkDwellClick(x, y);
    }

    checkDwellClick(x, y) {
        const now = performance.now();

        // Add current position to tracking array
        this.dwellPositions.push({ x, y, time: now });

        // Remove positions older than dwell duration
        this.dwellPositions = this.dwellPositions.filter(
            pos => now - pos.time < this.dwellDuration
        );

        // Check if we have enough samples
        if (this.dwellPositions.length < 10) {
            this.resetDwell();
            return;
        }

        // Calculate variance of positions
        const variance = this.calculatePositionVariance();

        if (variance < this.dwellRadius) {
            // User is dwelling
            if (!this.isDwelling) {
                this.isDwelling = true;
                this.dwellStartTime = now;
            }

            // Calculate dwell progress
            const dwellProgress = (now - this.dwellStartTime) / this.dwellDuration;
            this.renderDwellTimer(dwellProgress);

            // Trigger click if dwell complete
            if (dwellProgress >= 1.0 && now - this.lastClickTime > this.clickCooldown) {
                this.triggerClick(x, y);
                this.lastClickTime = now;
                this.resetDwell();
            }
        } else {
            this.resetDwell();
        }
    }

    calculatePositionVariance() {
        if (this.dwellPositions.length === 0) return Infinity;

        // Calculate center of positions
        const centerX = this.dwellPositions.reduce((sum, p) => sum + p.x, 0) / this.dwellPositions.length;
        const centerY = this.dwellPositions.reduce((sum, p) => sum + p.y, 0) / this.dwellPositions.length;

        // Calculate maximum distance from center
        const maxDistance = Math.max(...this.dwellPositions.map(p =>
            Math.sqrt(Math.pow(p.x - centerX, 2) + Math.pow(p.y - centerY, 2))
        ));

        return maxDistance;
    }

    renderDwellTimer(progress) {
        const ctx = this.dwellCtx;
        const centerX = this.dwellCanvas.width / 2;
        const centerY = this.dwellCanvas.height / 2;
        const radius = 14;

        ctx.clearRect(0, 0, this.dwellCanvas.width, this.dwellCanvas.height);

        // Draw progress arc
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, -Math.PI / 2, -Math.PI / 2 + (progress * 2 * Math.PI));
        ctx.strokeStyle = '#4ade80';
        ctx.lineWidth = 3;
        ctx.stroke();

        // Draw center dot
        ctx.beginPath();
        ctx.arc(centerX, centerY, 3, 0, 2 * Math.PI);
        ctx.fillStyle = '#4ade80';
        ctx.fill();
    }

    resetDwell() {
        this.isDwelling = false;
        this.dwellStartTime = null;
        this.dwellCtx.clearRect(0, 0, this.dwellCanvas.width, this.dwellCanvas.height);
    }

    triggerClick(x, y) {
        // Find element at cursor position
        const element = document.elementFromPoint(x, y);

        if (element && element !== this.cursorElement) {
            // Create and dispatch click event
            const clickEvent = new MouseEvent('click', {
                view: window,
                bubbles: true,
                cancelable: true,
                clientX: x,
                clientY: y
            });

            element.dispatchEvent(clickEvent);

            // Visual feedback
            this.showClickFeedback(x, y);
        }
    }

    showClickFeedback(x, y) {
        const feedback = document.createElement('div');
        feedback.style.position = 'fixed';
        feedback.style.left = `${x}px`;
        feedback.style.top = `${y}px`;
        feedback.style.width = '40px';
        feedback.style.height = '40px';
        feedback.style.border = '3px solid #4ade80';
        feedback.style.borderRadius = '50%';
        feedback.style.transform = 'translate(-50%, -50%)';
        feedback.style.pointerEvents = 'none';
        feedback.style.animation = 'clickPulse 0.4s ease-out';
        feedback.style.zIndex = '10000';

        document.body.appendChild(feedback);

        setTimeout(() => feedback.remove(), 400);

        // Add CSS animation if not already present
        if (!document.getElementById('click-feedback-style')) {
            const style = document.createElement('style');
            style.id = 'click-feedback-style';
            style.textContent = `
                @keyframes clickPulse {
                    0% { transform: translate(-50%, -50%) scale(0.5); opacity: 1; }
                    100% { transform: translate(-50%, -50%) scale(1.5); opacity: 0; }
                }
            `;
            document.head.appendChild(style);
        }
    }

    toggle() {
        this.isVisible = !this.isVisible;
        this.cursorElement.style.opacity = this.isVisible ? '1' : '0';
    }

    setPosition(x, y) {
        this.update(x, y);
    }
}
