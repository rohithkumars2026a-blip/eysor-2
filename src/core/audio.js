/**
 * JARVIS Audio & Speech Guidance System for EYSOR
 * Provides sophisticated British AI voice assistance and futuristic sound effects
 */
export class AudioGuide {
    constructor() {
        this.isMuted = false;
        this.audioContext = null;
        this.utterance = null;
        this.selectedVoice = null;

        // Check for Speech Synthesis API
        this.hasSpeech = 'speechSynthesis' in window;

        // Load mute state from localStorage
        this.isMuted = localStorage.getItem('eysor_audio_muted') === 'true';
    }

    init() {
        // Initialize Web Audio Context for JARVIS holographic UI sound effects
        if ('AudioContext' in window || 'webkitAudioContext' in window) {
            try {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            } catch (e) {
                console.warn('Web Audio API not supported:', e);
            }
        }

        if (this.hasSpeech) {
            this.updateVoice();
            if (window.speechSynthesis.onvoiceschanged !== undefined) {
                window.speechSynthesis.onvoiceschanged = () => this.updateVoice();
            }
        }

        console.log(`🎙️ JARVIS Audio Core initialized (Muted: ${this.isMuted})`);
    }

    updateVoice() {
        if (!this.hasSpeech) return;
        const voices = window.speechSynthesis.getVoices();
        if (!voices || voices.length === 0) return;

        // Search for British / UK Male voices for JARVIS persona
        this.selectedVoice =
            voices.find(v => (v.name.includes('Google UK English Male') || v.name.includes('Daniel') || v.name.includes('George') || v.name.includes('Oliver') || v.name.includes('Arthur') || v.name.includes('Brian'))) ||
            voices.find(v => (v.lang === 'en-GB' || v.lang === 'en-UK' || v.lang.startsWith('en-GB')) && (v.name.toLowerCase().includes('male') || !v.name.toLowerCase().includes('female'))) ||
            voices.find(v => v.lang === 'en-GB' || v.lang === 'en-UK' || v.lang.startsWith('en-GB')) ||
            voices.find(v => v.lang.startsWith('en') && v.name.toLowerCase().includes('male')) ||
            null;

        if (this.selectedVoice) {
            console.log(`🤖 JARVIS Voice Assigned: ${this.selectedVoice.name} (${this.selectedVoice.lang})`);
        }
    }

    speak(text, options = {}) {
        if (this.isMuted || !this.hasSpeech) return;

        // Cancel any ongoing utterance for immediate response
        if (window.speechSynthesis.speaking) {
            window.speechSynthesis.cancel();
        }

        if (!this.selectedVoice) {
            this.updateVoice();
        }

        this.utterance = new SpeechSynthesisUtterance(text);
        // JARVIS speech characteristics: articulate, measured, polished baritone
        this.utterance.rate = options.rate || 0.94;
        this.utterance.pitch = options.pitch || 0.88;
        this.utterance.volume = options.volume || 0.9;
        this.utterance.lang = 'en-GB';

        if (this.selectedVoice) {
            this.utterance.voice = this.selectedVoice;
        }

        window.speechSynthesis.speak(this.utterance);
    }

    /**
     * High-tech futuristic holographic ping (Iron Man HUD style)
     */
    playHoloPing(freq = 880, duration = 150) {
        if (this.isMuted || !this.audioContext) return;

        try {
            if (this.audioContext.state === 'suspended') {
                this.audioContext.resume();
            }

            const osc1 = this.audioContext.createOscillator();
            const osc2 = this.audioContext.createOscillator();
            const gainNode = this.audioContext.createGain();

            osc1.type = 'sine';
            osc1.frequency.setValueAtTime(freq, this.audioContext.currentTime);
            osc1.frequency.exponentialRampToValueAtTime(freq * 1.5, this.audioContext.currentTime + duration / 1000);

            osc2.type = 'triangle';
            osc2.frequency.setValueAtTime(freq * 2, this.audioContext.currentTime);

            gainNode.gain.setValueAtTime(0.15, this.audioContext.currentTime);
            gainNode.gain.exponentialRampToValueAtTime(0.001, this.audioContext.currentTime + duration / 1000);

            osc1.connect(gainNode);
            osc2.connect(gainNode);
            gainNode.connect(this.audioContext.destination);

            osc1.start(this.audioContext.currentTime);
            osc2.start(this.audioContext.currentTime);
            osc1.stop(this.audioContext.currentTime + duration / 1000);
            osc2.stop(this.audioContext.currentTime + duration / 1000);
        } catch (e) {
            // Audio context safely caught
        }
    }

    playBeep(frequency = 520, duration = 120) {
        this.playHoloPing(frequency, duration);
    }

    playSuccess() {
        if (this.isMuted || !this.audioContext) return;

        // JARVIS system online chime (Harmonic triad)
        [523.25, 659.25, 783.99, 1046.50].forEach((freq, i) => {
            setTimeout(() => this.playHoloPing(freq, 220), i * 90);
        });
    }

    playError() {
        if (this.isMuted || !this.audioContext) return;

        this.playHoloPing(240, 250);
        setTimeout(() => this.playHoloPing(180, 300), 140);
    }

    playProgress() {
        if (this.isMuted || !this.audioContext) return;
        this.playHoloPing(720, 100);
    }

    playBreak() {
        if (this.isMuted || !this.audioContext) return;

        // Soothing descending chord for relaxation
        [659.25, 523.25, 392.00].forEach((freq, i) => {
            setTimeout(() => this.playHoloPing(freq, 300), i * 140);
        });
    }

    // JARVIS Persona Guidance Scripts
    welcomeMessage() {
        this.speak("Good day, sir. JARVIS online. All systems initialized. Shall we commence the ocular calibration sequence?");
    }

    startCalibration(totalPoints) {
        this.speak(`Initiating ocular calibration protocol. Please direct your gaze at each luminous indicator as it appears. We shall process ${totalPoints} calibration nodes with scheduled relaxation intervals.`);
    }

    nextPoint(currentPoint, totalPoints) {
        if (currentPoint % 3 === 0) {
            this.speak(`Calibration node ${currentPoint} of ${totalPoints}. Optical telemetry locked.`);
        }
        this.playProgress();
    }

    eyeRestBreak(seconds) {
        this.speak(`Optical fatigue protection protocol engaged, sir. I recommend resting your eyes for ${seconds} seconds. Please close your eyes and relax. I shall notify you when ready.`);
        this.playBreak();
    }

    resumeCalibration() {
        this.speak("Rest period concluded, sir. Resuming calibration sequence. Target acquired.");
        this.playHoloPing(880, 150);
    }

    calibrationComplete() {
        this.speak("Ocular calibration sequence completed successfully, sir. Now initiating TITAN neural network training. This will take approximately 3 to 5 minutes.");
        this.playSuccess();
    }

    trainingProgress(modelNumber, totalModels) {
        if (modelNumber % 5 === 0) {
            this.speak(`Neural model ${modelNumber} of ${totalModels} converged. All telemetry nominal.`);
        }
    }

    trainingComplete() {
        this.speak("Neural network optimization complete, sir. The TITAN AI system is now operating at peak precision. Sub-pixel tracking active.");
        this.playSuccess();
    }

    verificationStart() {
        this.speak("Commencing precision verification protocol. Please track each validation coordinate, sir.");
    }

    verificationComplete(accuracy) {
        const quality = accuracy < 0.5 ? "Flawless, sub-half-pixel accuracy achieved" : accuracy < 1.0 ? "Exceptional precision" : "Highly accurate";
        this.speak(`Verification protocol concluded, sir. Mean variance is ${accuracy.toFixed(2)} pixels. ${quality}.`);
        this.playSuccess();
    }

    toggle() {
        this.isMuted = !this.isMuted;
        localStorage.setItem('eysor_audio_muted', this.isMuted.toString());

        if (!this.isMuted) {
            this.speak("Audio systems online, sir.");
        }

        return this.isMuted;
    }

    getMutedState() {
        return this.isMuted;
    }

    stop() {
        if (this.hasSpeech && window.speechSynthesis.speaking) {
            window.speechSynthesis.cancel();
        }
    }
}
