import { FaceMesh } from '@mediapipe/face_mesh';
import { Camera } from '@mediapipe/camera_utils';

export class VisionEngine {
    constructor(wearsGlasses = false) {
        this.faceMesh = null;
        this.camera = null;
        this.videoElement = null;
        this.latestResults = null;
        this.latestGaze = null;
        this.wearsGlasses = wearsGlasses;

        // Iris landmark indices (MediaPipe Face Mesh)
        this.LEFT_IRIS = [468, 469, 470, 471, 472];
        this.RIGHT_IRIS = [473, 474, 475, 476, 477];
        this.LEFT_EYE_CORNERS = [33, 133]; // inner, outer
        this.RIGHT_EYE_CORNERS = [362, 263];
        this.LEFT_EYE_TOP_BOTTOM = [159, 145]; // upper, lower eyelid
        this.RIGHT_EYE_TOP_BOTTOM = [386, 374];

        // Reference points for head pose estimation
        this.POSE_LANDMARKS = [1, 33, 61, 199, 263, 291]; // nose, eyes, chin

        // Glasses-specific landmarks for glare detection
        this.FOREHEAD = [10, 338, 297, 332, 284];
        this.CHEEKBONES = [234, 454];
    }

    async init() {
        try {
            this.videoElement = document.getElementById('webcam');

            if (!this.videoElement) {
                throw new Error('Video element not found');
            }

            console.log('Initializing MediaPipe Face Mesh...');

            // Initialize MediaPipe Face Mesh
            this.faceMesh = new FaceMesh({
                locateFile: (file) => {
                    return `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`;
                }
            });

            this.faceMesh.setOptions({
                maxNumFaces: 1,
                refineLandmarks: true,
                minDetectionConfidence: 0.3,
                minTrackingConfidence: 0.3
            });

            this.faceMesh.onResults((results) => this.onResults(results));

            console.log('Requesting camera access...');

            // Initialize camera
            this.camera = new Camera(this.videoElement, {
                onFrame: async () => {
                    if (this.faceMesh) {
                        await this.faceMesh.send({ image: this.videoElement });
                    }
                },
                width: 1280,
                height: 720
            });

            await this.camera.start();
            console.log('Camera started successfully');

        } catch (error) {
            console.error('Vision initialization error:', error);
            throw new Error(`Camera/MediaPipe initialization failed: ${error.message}`);
        }
    }

    onResults(results) {
        this.latestResults = results;

        if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
            const landmarks = results.multiFaceLandmarks[0];
            const features = this.extractFeatures(landmarks);
            const gaze = this.estimateGaze(features);

            this.latestGaze = {
                x: gaze.x,
                y: gaze.y,
                features: features
            };
        }
    }

    extractFeatures(landmarks) {
        // Extract iris centers
        const leftIris = this.getIrisCenter(landmarks, this.LEFT_IRIS);
        const rightIris = this.getIrisCenter(landmarks, this.RIGHT_IRIS);

        // Extract eye corners
        const leftInner = landmarks[this.LEFT_EYE_CORNERS[0]];
        const leftOuter = landmarks[this.LEFT_EYE_CORNERS[1]];
        const rightInner = landmarks[this.RIGHT_EYE_CORNERS[0]];
        const rightOuter = landmarks[this.RIGHT_EYE_CORNERS[1]];

        // Extract eyelid positions for aperture calculation
        const leftTop = landmarks[this.LEFT_EYE_TOP_BOTTOM[0]];
        const leftBottom = landmarks[this.LEFT_EYE_TOP_BOTTOM[1]];
        const rightTop = landmarks[this.RIGHT_EYE_TOP_BOTTOM[0]];
        const rightBottom = landmarks[this.RIGHT_EYE_TOP_BOTTOM[1]];

        // Calculate eye aperture (openness)
        const leftAperture = Math.abs(leftTop.y - leftBottom.y);
        const rightAperture = Math.abs(rightTop.y - rightBottom.y);

        // Calculate iris ratios (normalized position within eye)
        const leftRatioX = (leftIris.x - leftInner.x) / (leftOuter.x - leftInner.x);
        const leftRatioY = leftIris.y;
        const rightRatioX = (rightIris.x - rightOuter.x) / (rightInner.x - rightOuter.x);
        const rightRatioY = rightIris.y;

        // Calculate head pose (simplified using landmark positions)
        const headPose = this.estimateHeadPose(landmarks);

        const features = {
            leftIrisRatioX: leftRatioX,
            leftIrisRatioY: leftRatioY,
            rightIrisRatioX: rightRatioX,
            rightIrisRatioY: rightRatioY,
            avgIrisRatioX: (leftRatioX + rightRatioX) / 2,
            avgIrisRatioY: (leftRatioY + rightRatioY) / 2,
            headPitch: headPose.pitch,
            headYaw: headPose.yaw,
            headRoll: headPose.roll,
            leftAperture: leftAperture,
            rightAperture: rightAperture,
            avgAperture: (leftAperture + rightAperture) / 2
        };

        // Add glasses-specific features if enabled
        if (this.wearsGlasses) {
            const glassesFeatures = this.extractGlassesFeatures(landmarks, leftIris, rightIris);
            Object.assign(features, glassesFeatures);
        }

        return features;
    }

    extractGlassesFeatures(landmarks, leftIris, rightIris) {
        // Calculate 3D depth variance to detect reflections/glare
        const leftIrisDepth = leftIris.z || 0;
        const rightIrisDepth = rightIris.z || 0;

        // Calculate distance from iris to frame edges (approximation)
        const leftEyeWidth = Math.abs(landmarks[this.LEFT_EYE_CORNERS[1]].x - landmarks[this.LEFT_EYE_CORNERS[0]].x);
        const rightEyeWidth = Math.abs(landmarks[this.RIGHT_EYE_CORNERS[0]].x - landmarks[this.RIGHT_EYE_CORNERS[1]].x);

        // Normalized iris position relative to eye height
        const leftEyeHeight = Math.abs(landmarks[this.LEFT_EYE_TOP_BOTTOM[0]].y - landmarks[this.LEFT_EYE_TOP_BOTTOM[1]].y);
        const rightEyeHeight = Math.abs(landmarks[this.RIGHT_EYE_TOP_BOTTOM[0]].y - landmarks[this.RIGHT_EYE_TOP_BOTTOM[1]].y);

        // Calculate normalized vertical position
        const leftVerticalPos = (leftIris.y - landmarks[this.LEFT_EYE_TOP_BOTTOM[1]].y) / leftEyeHeight;
        const rightVerticalPos = (rightIris.y - landmarks[this.RIGHT_EYE_TOP_BOTTOM[1]].y) / rightEyeHeight;

        return {
            leftIrisDepth: leftIrisDepth,
            rightIrisDepth: rightIrisDepth,
            depthDifference: Math.abs(leftIrisDepth - rightIrisDepth),
            leftEyeWidth: leftEyeWidth,
            rightEyeWidth: rightEyeWidth,
            leftEyeHeight: leftEyeHeight,
            rightEyeHeight: rightEyeHeight,
            leftVerticalPos: leftVerticalPos,
            rightVerticalPos: rightVerticalPos,
            avgVerticalPos: (leftVerticalPos + rightVerticalPos) / 2
        };
    }

    getIrisCenter(landmarks, irisIndices) {
        let sumX = 0, sumY = 0, sumZ = 0;

        for (const idx of irisIndices) {
            sumX += landmarks[idx].x;
            sumY += landmarks[idx].y;
            sumZ += landmarks[idx].z || 0;
        }

        return {
            x: sumX / irisIndices.length,
            y: sumY / irisIndices.length,
            z: sumZ / irisIndices.length
        };
    }

    estimateHeadPose(landmarks) {
        // Simplified head pose estimation using key landmark positions
        const noseTip = landmarks[1];
        const leftEye = landmarks[33];
        const rightEye = landmarks[263];
        const chin = landmarks[199];

        // Calculate yaw (horizontal head rotation)
        const eyeMidX = (leftEye.x + rightEye.x) / 2;
        const yaw = (noseTip.x - eyeMidX) * 2; // Normalized approx

        // Calculate pitch (vertical head rotation)
        const eyeMidY = (leftEye.y + rightEye.y) / 2;
        const pitch = (noseTip.y - eyeMidY) * 2;

        // Calculate roll (head tilt)
        const roll = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x);

        return {
            pitch: pitch,
            yaw: yaw,
            roll: roll
        };
    }

    estimateGaze(features) {
        // Convert iris ratios to screen coordinates (initial estimate)
        // This will be refined by the ML corrector after calibration
        const screenWidth = window.innerWidth;
        const screenHeight = window.innerHeight;

        // Apply head pose compensation
        let gazeX = features.avgIrisRatioX;
        let gazeY = features.avgIrisRatioY;

        // Compensate for head yaw/pitch
        gazeX -= features.headYaw * 0.3;
        gazeY -= features.headPitch * 0.3;

        return {
            x: gazeX * screenWidth,
            y: gazeY * screenHeight
        };
    }

    getLatestGaze() {
        return this.latestGaze;
    }

    getVideoElement() {
        return this.videoElement;
    }
}
