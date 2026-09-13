import * as tf from '@tensorflow/tfjs';
import { GazeFilter } from '../core/filtering.js';

/**
 * TITAN Fast-Init System
 * Quick-loading core models with lazy initialization of advanced models
 */
export class TitanCorrector {
    constructor(wearsGlasses = false) {
        this.wearsGlasses = wearsGlasses;

        // Core models only (fast init)
        this.models = {
            primary: null,      // Primary correction model
            residual: null,     // Error correction
            ensemble: null      // Meta-learner
        };

        // Advanced models loaded lazily during training
        this.advancedModels = null;

        // Kalman Filter Matrix
        this.kalman = this.initKalmanFilter();

        // Quad-Stage Filtering Pipeline
        this.filters = {
            stage1: new GazeFilter(
                wearsGlasses ? 0.25 : 0.4,
                wearsGlasses ? 0.03 : 0.02,
                wearsGlasses ? 0.35 : 0.5
            ),
            stage2: new GazeFilter(
                wearsGlasses ? 0.18 : 0.25,
                wearsGlasses ? 0.035 : 0.025,
                wearsGlasses ? 0.22 : 0.3
            ),
            stage3: new GazeFilter(
                wearsGlasses ? 0.12 : 0.15,
                wearsGlasses ? 0.04 : 0.03,
                wearsGlasses ? 0.18 : 0.25
            ),
            stage4: new GazeFilter(
                wearsGlasses ? 0.08 : 0.1,
                wearsGlasses ? 0.04 : 0.03,
                wearsGlasses ? 0.18 : 0.25
            )
        };

        this.isModelTrained = false;
        this.featureMean = null;
        this.featureStd = null;
        this.modelId = wearsGlasses ? 'titan-gaze-glasses-v4' : 'titan-gaze-v4';

        // Temporal buffering
        this.temporalBuffer = [];
        this.temporalWindowSize = 15;
    }

    initKalmanFilter() {
        return {
            state: [0, 0, 0, 0], // [x, y, vx, vy]
            P: [
                [500, 0, 0, 0],
                [0, 500, 0, 0],
                [0, 0, 500, 0],
                [0, 0, 0, 500]
            ],
            Q: [
                [0.005, 0, 0, 0],
                [0, 0.005, 0, 0],
                [0, 0, 0.05, 0],
                [0, 0, 0, 0.05]
            ],
            R: [[0.2, 0], [0, 0.2]],
            F: [
                [1, 0, 0.016, 0],
                [0, 1, 0, 0.016],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ],
            H: [[1, 0, 0, 0], [0, 1, 0, 0]]
        };
    }

    async init() {
        console.log(`⚡ Quick-initializing TITAN Core (${this.wearsGlasses ? 'Glasses Mode' : 'Standard Mode'})...`);

        try {
            await this.loadModels();
            console.log('✅ Loaded existing TITAN models from cache');
        } catch (error) {
            console.log('⚡ Creating minimal model for instant startup...');
            this.createMinimalModel();
            console.log('✅ TITAN ready! (Full models will build during calibration)');
        }
    }

    createMinimalModel() {
        const inputSize = this.wearsGlasses ? 21 : 12;

        // Single lightweight model for instant startup
        this.models.primary = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [inputSize],
                    units: 64,
                    activation: 'relu',
                    kernelInitializer: 'heNormal'
                }),
                tf.layers.dense({ units: 32, activation: 'relu' }),
                tf.layers.dense({ units: 2, activation: 'linear' })
            ]
        });
        this.models.primary.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError'
        });

        console.log('✅ Minimal model created (instant startup)');
    }

    async buildFullModels() {
        console.log('🔧 Building full TITAN ensemble in background...');
        const inputSize = this.wearsGlasses ? 21 : 12;

        // Build residual and ensemble models
        this.models.residual = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [inputSize],
                    units: 96,
                    activation: 'relu',
                    kernelInitializer: 'heNormal'
                }),
                tf.layers.dropout({ rate: 0.25 }),
                tf.layers.dense({ units: 48, activation: 'relu' }),
                tf.layers.dense({ units: 2, activation: 'linear' })
            ]
        });
        this.models.residual.compile({
            optimizer: tf.train.adam(0.0008),
            loss: 'meanSquaredError'
        });

        this.models.ensemble = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [4],
                    units: 32,
                    activation: 'relu'
                }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({ units: 16, activation: 'relu' }),
                tf.layers.dense({ units: 2, activation: 'linear' })
            ]
        });
        this.models.ensemble.compile({
            optimizer: tf.train.adam(0.0005),
            loss: 'meanSquaredError'
        });

        console.log('✅ Full TITAN ensemble ready');
    }

    async loadModels() {
        const loadPromises = Object.keys(this.models).map(async (key) => {
            const modelPath = `indexeddb://${this.modelId}-${key}`;
            this.models[key] = await tf.loadLayersModel(modelPath);
        });

        await Promise.all(loadPromises);
        this.isModelTrained = true;
    }

    createFastModels() {
        const inputSize = this.wearsGlasses ? 21 : 12;

        // 1. Primary correction model (lightweight but effective)
        this.models.primary = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [inputSize],
                    units: 128,
                    activation: 'relu',
                    kernelInitializer: 'heNormal'
                }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({ units: 64, activation: 'relu' }),
                tf.layers.dense({ units: 32, activation: 'relu' }),
                tf.layers.dense({ units: 2, activation: 'linear' })
            ]
        });
        this.models.primary.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'meanSquaredError'
        });

        // 2. Residual error corrector
        this.models.residual = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [inputSize],
                    units: 96,
                    activation: 'relu',
                    kernelInitializer: 'heNormal'
                }),
                tf.layers.dropout({ rate: 0.25 }),
                tf.layers.dense({ units: 48, activation: 'relu' }),
                tf.layers.dense({ units: 2, activation: 'linear' })
            ]
        });
        this.models.residual.compile({
            optimizer: tf.train.adam(0.0008),
            loss: 'meanSquaredError'
        });

        // 3. Ensemble meta-learner
        this.models.ensemble = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [4], // Takes outputs from primary + residual
                    units: 32,
                    activation: 'relu'
                }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({ units: 16, activation: 'relu' }),
                tf.layers.dense({ units: 2, activation: 'linear' })
            ]
        });
        this.models.ensemble.compile({
            optimizer: tf.train.adam(0.0005),
            loss: 'meanSquaredError'
        });

        console.log('✅ 3 Core models created (fast initialization complete)');
    }

    async train(calibrationData, audioGuide = null) {
        if (!calibrationData || calibrationData.length === 0) {
            throw new Error('No calibration data provided');
        }

        console.log(`🚀 Training TITAN on ${calibrationData.length} samples...`);

        // Build full models if not already built
        if (!this.models.residual || !this.models.ensemble) {
            await this.buildFullModels();
        }

        const { inputs, outputs } = this.prepareTrainingData(calibrationData);
        this.computeNormalization(inputs);
        const normalizedInputs = this.normalizeFeatures(inputs);

        const xTensor = tf.tensor2d(normalizedInputs);
        const yTensor = tf.tensor2d(outputs);

        const epochs = this.wearsGlasses ? 60 : 50;
        const batchSize = 8;

        // Train primary model
        console.log('Training primary correction model...');
        if (audioGuide) {
            audioGuide.speak("Training primary neural network, sir.");
        }

        await this.models.primary.fit(xTensor, yTensor, {
            epochs: epochs,
            batchSize: batchSize,
            verbose: 0,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    if ((epoch + 1) % 15 === 0) {
                        console.log(`  Primary: Epoch ${epoch + 1}/${epochs} - Loss: ${logs.loss.toFixed(4)}`);
                    }
                }
            }
        });

        // Get residuals for residual model
        const primaryPredictions = this.models.primary.predict(xTensor);
        const residuals = tf.sub(yTensor, primaryPredictions);

        // Train residual model
        console.log('Training residual error corrector...');
        if (audioGuide) {
            audioGuide.speak("Training residual error corrector.");
        }

        await this.models.residual.fit(xTensor, residuals, {
            epochs: Math.round(epochs * 0.7),
            batchSize: batchSize,
            verbose: 0,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    if ((epoch + 1) % 10 === 0) {
                        console.log(`  Residual: Epoch ${epoch + 1}/${Math.round(epochs * 0.7)} - Loss: ${logs.loss.toFixed(4)}`);
                    }
                }
            }
        });

        // Create ensemble training data
        const residualPredictions = this.models.residual.predict(xTensor);
        const ensembleInputs = tf.concat([primaryPredictions, residualPredictions], 1);

        // Train ensemble
        console.log('Training ensemble meta-learner...');
        if (audioGuide) {
            audioGuide.speak("Finalizing ensemble meta-learner.");
        }

        await this.models.ensemble.fit(ensembleInputs, yTensor, {
            epochs: Math.round(epochs * 0.5),
            batchSize: batchSize,
            verbose: 0,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    if ((epoch + 1) % 8 === 0) {
                        console.log(`  Ensemble: Epoch ${epoch + 1}/${Math.round(epochs * 0.5)} - Loss: ${logs.loss.toFixed(4)}`);
                    }
                }
            }
        });

        // Cleanup
        primaryPredictions.dispose();
        residuals.dispose();
        residualPredictions.dispose();
        ensembleInputs.dispose();
        xTensor.dispose();
        yTensor.dispose();

        this.isModelTrained = true;
        console.log('✅ TITAN training complete!');

        // Save models
        await this.saveModels();
    }

    prepareTrainingData(calibrationData) {
        const inputs = [];
        const outputs = [];

        for (const sample of calibrationData) {
            const features = this.extractFeatures(sample.features);
            inputs.push(features);
            outputs.push([sample.targetX, sample.targetY]);
        }

        return { inputs, outputs };
    }

    extractFeatures(features) {
        if (this.wearsGlasses) {
            return [
                features.leftIrisRatioX,
                features.leftIrisRatioY,
                features.rightIrisRatioX,
                features.rightIrisRatioY,
                features.avgIrisRatioX,
                features.avgIrisRatioY,
                features.headPitch,
                features.headYaw,
                features.headRoll,
                features.leftAperture,
                features.rightAperture,
                features.avgAperture,
                features.leftIrisDepth || 0,
                features.rightIrisDepth || 0,
                features.depthDifference || 0,
                features.leftEyeWidth || 0,
                features.rightEyeWidth || 0,
                features.leftEyeHeight || 0,
                features.rightEyeHeight || 0,
                features.leftVerticalPos || 0,
                features.rightVerticalPos || 0
            ];
        } else {
            return [
                features.leftIrisRatioX,
                features.leftIrisRatioY,
                features.rightIrisRatioX,
                features.rightIrisRatioY,
                features.avgIrisRatioX,
                features.avgIrisRatioY,
                features.headPitch,
                features.headYaw,
                features.headRoll,
                features.leftAperture,
                features.rightAperture,
                features.avgAperture
            ];
        }
    }

    computeNormalization(inputs) {
        const features = inputs[0].length;
        this.featureMean = new Array(features).fill(0);
        this.featureStd = new Array(features).fill(0);

        for (let f = 0; f < features; f++) {
            let sum = 0;
            for (const input of inputs) {
                sum += input[f];
            }
            this.featureMean[f] = sum / inputs.length;
        }

        for (let f = 0; f < features; f++) {
            let sumSq = 0;
            for (const input of inputs) {
                sumSq += Math.pow(input[f] - this.featureMean[f], 2);
            }
            this.featureStd[f] = Math.sqrt(sumSq / inputs.length) + 1e-8;
        }
    }

    normalizeFeatures(inputs) {
        return inputs.map(input =>
            input.map((val, i) => (val - this.featureMean[i]) / this.featureStd[i])
        );
    }

    correct(rawX, rawY, features) {
        if (!this.isModelTrained) {
            return { x: rawX, y: rawY };
        }

        const featureArray = this.extractFeatures(features);
        const normalized = featureArray.map((val, i) =>
            (val - this.featureMean[i]) / this.featureStd[i]
        );

        const inputTensor = tf.tensor2d([normalized]);

        let correctedX, correctedY;

        // Use ensemble if available, otherwise just primary
        if (this.models.residual && this.models.ensemble) {
            // Primary correction
            const primaryPred = this.models.primary.predict(inputTensor);

            // Residual correction
            const residualPred = this.models.residual.predict(inputTensor);

            // Ensemble correction
            const ensembleInput = tf.concat([primaryPred, residualPred], 1);
            const ensemblePred = this.models.ensemble.predict(ensembleInput);

            [correctedX, correctedY] = ensemblePred.dataSync();

            // Cleanup
            primaryPred.dispose();
            residualPred.dispose();
            ensembleInput.dispose();
            ensemblePred.dispose();
        } else {
            // Use primary model only (during initial startup)
            const primaryPred = this.models.primary.predict(inputTensor);
            [correctedX, correctedY] = primaryPred.dataSync();
            primaryPred.dispose();
        }

        inputTensor.dispose();

        // Apply Kalman filter
        const kalmanFiltered = this.kalmanFilter(correctedX, correctedY);

        // Apply quad-stage OneEuro filtering
        let finalX = kalmanFiltered.x;
        let finalY = kalmanFiltered.y;

        finalX = this.filters.stage1.filter(finalX);
        finalY = this.filters.stage1.filter(finalY);

        finalX = this.filters.stage2.filter(finalX);
        finalY = this.filters.stage2.filter(finalY);

        finalX = this.filters.stage3.filter(finalX);
        finalY = this.filters.stage3.filter(finalY);

        finalX = this.filters.stage4.filter(finalX);
        finalY = this.filters.stage4.filter(finalY);

        return { x: finalX, y: finalY };
    }

    kalmanFilter(measuredX, measuredY) {
        const F = this.kalman.F;
        const H = this.kalman.H;
        const Q = this.kalman.Q;
        const R = this.kalman.R;
        let state = this.kalman.state;
        let P = this.kalman.P;

        // Predict
        state = this.matrixVectorMult(F, state);
        P = this.matrixAdd(this.matrixMult(F, this.matrixMult(P, this.transpose(F))), Q);

        // Update
        const y = [measuredX - (H[0][0] * state[0]), measuredY - (H[1][1] * state[1])];
        const S = this.matrixAdd(this.matrixMult(H, this.matrixMult(P, this.transpose(H))), R);
        const K = this.matrixMult(this.matrixMult(P, this.transpose(H)), this.matrixInv2x2(S));

        state = this.vectorAdd(state, this.matrixVectorMult(K, y));
        const I = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];
        P = this.matrixMult(this.matrixSub(I, this.matrixMult(K, H)), P);

        this.kalman.state = state;
        this.kalman.P = P;

        return { x: state[0], y: state[1] };
    }

    matrixMult(A, B) {
        const result = [];
        for (let i = 0; i < A.length; i++) {
            result[i] = [];
            for (let j = 0; j < B[0].length; j++) {
                let sum = 0;
                for (let k = 0; k < A[0].length; k++) {
                    sum += A[i][k] * B[k][j];
                }
                result[i][j] = sum;
            }
        }
        return result;
    }

    matrixVectorMult(A, v) {
        const result = [];
        for (let i = 0; i < A.length; i++) {
            let sum = 0;
            for (let j = 0; j < A[0].length; j++) {
                sum += A[i][j] * v[j];
            }
            result[i] = sum;
        }
        return result;
    }

    matrixAdd(A, B) {
        return A.map((row, i) => row.map((val, j) => val + B[i][j]));
    }

    matrixSub(A, B) {
        return A.map((row, i) => row.map((val, j) => val - B[i][j]));
    }

    vectorAdd(a, b) {
        return a.map((val, i) => val + b[i]);
    }

    transpose(A) {
        return A[0].map((_, i) => A.map(row => row[i]));
    }

    matrixInv2x2(A) {
        const det = A[0][0] * A[1][1] - A[0][1] * A[1][0];
        return [
            [A[1][1] / det, -A[0][1] / det],
            [-A[1][0] / det, A[0][0] / det]
        ];
    }

    async saveModels() {
        console.log('💾 Saving TITAN models...');
        const savePromises = Object.entries(this.models).map(async ([key, model]) => {
            const modelPath = `indexeddb://${this.modelId}-${key}`;
            await model.save(modelPath);
        });

        await Promise.all(savePromises);
        console.log('✅ Models saved to IndexedDB');
    }
}
