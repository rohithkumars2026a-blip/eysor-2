import * as tf from '@tensorflow/tfjs';
import { GazeFilter } from '../core/filtering.js';

export class AdaptiveCorrector {
    constructor(wearsGlasses = false) {
        this.model = null;
        this.wearsGlasses = wearsGlasses;

        // Different filter parameters for glasses wearers (more aggressive filtering)
        if (wearsGlasses) {
            this.filter = new GazeFilter(0.8, 0.01, 0.8); // Lower minCutoff, higher beta for stability
        } else {
            this.filter = new GazeFilter(1.0, 0.007, 1.0);
        }

        this.isModelTrained = false;

        // Feature normalization parameters (learned during training)
        this.featureMean = null;
        this.featureStd = null;

        // Model identifier based on glasses mode
        this.modelId = wearsGlasses ? 'gaze-model-glasses-v1' : 'gaze-model-v1';
    }

    async init() {
        // Try to load existing model from IndexedDB
        try {
            this.model = await tf.loadLayersModel(`indexeddb://${this.modelId}`);
            this.isModelTrained = true;
            console.log(`Loaded existing ${this.wearsGlasses ? 'glasses' : 'standard'} model from IndexedDB`);
        } catch (error) {
            console.log('No existing model found, will train new model');
            this.createModel();
        }
    }

    createModel() {
        // Enhanced model architecture for glasses wearers
        const inputSize = this.wearsGlasses ? 21 : 12; // More features for glasses mode
        const hiddenUnits1 = this.wearsGlasses ? 128 : 64; // Larger network for glasses
        const hiddenUnits2 = this.wearsGlasses ? 64 : 32;
        const hiddenUnits3 = this.wearsGlasses ? 32 : null; // Extra layer for glasses

        const layers = [
            tf.layers.dense({
                inputShape: [inputSize],
                units: hiddenUnits1,
                activation: 'relu',
                kernelInitializer: 'heNormal'
            }),
            tf.layers.dropout({ rate: 0.25 }),
            tf.layers.dense({
                units: hiddenUnits2,
                activation: 'relu',
                kernelInitializer: 'heNormal'
            }),
            tf.layers.dropout({ rate: 0.15 })
        ];

        // Add extra layer for glasses mode
        if (this.wearsGlasses && hiddenUnits3) {
            layers.push(
                tf.layers.dense({
                    units: hiddenUnits3,
                    activation: 'relu',
                    kernelInitializer: 'heNormal'
                }),
                tf.layers.dropout({ rate: 0.1 })
            );
        }

        // Output layer
        layers.push(
            tf.layers.dense({
                units: 2, // Output: corrected (x, y)
                activation: 'linear'
            })
        );

        this.model = tf.sequential({ layers });

        this.model.compile({
            optimizer: tf.train.adam(0.0008), // Slightly lower learning rate for stability
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        console.log(`Created ${this.wearsGlasses ? 'enhanced glasses' : 'standard'} model with ${inputSize} input features`);
    }

    async train(calibrationData) {
        if (!calibrationData || calibrationData.length === 0) {
            throw new Error('No calibration data provided');
        }

        console.log(`Training on ${calibrationData.length} samples`);

        // Prepare training data
        const { inputs, outputs } = this.prepareTrainingData(calibrationData);

        // Normalize features
        this.computeNormalization(inputs);
        const normalizedInputs = this.normalizeFeatures(inputs);

        // Convert to tensors
        const xs = tf.tensor2d(normalizedInputs);
        const ys = tf.tensor2d(outputs);

        // Train model with more epochs for glasses mode
        const epochs = this.wearsGlasses ? 80 : 50;
        const batchSize = this.wearsGlasses ? 16 : 32; // Smaller batches for better convergence

        await this.model.fit(xs, ys, {
            epochs: epochs,
            batchSize: batchSize,
            validationSplit: 0.15,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    if (epoch % 10 === 0) {
                        console.log(`Epoch ${epoch}: loss = ${logs.loss.toFixed(4)}, mae = ${logs.mae.toFixed(2)}px`);
                    }
                }
            }
        });

        // Clean up tensors
        xs.dispose();
        ys.dispose();

        // Save model to IndexedDB
        await this.model.save(`indexeddb://${this.modelId}`);
        this.isModelTrained = true;

        console.log(`${this.wearsGlasses ? 'Glasses' : 'Standard'} model training complete and saved`);
    }

    prepareTrainingData(calibrationData) {
        const inputs = [];
        const outputs = [];

        for (const sample of calibrationData) {
            const features = sample.features;

            let inputVector;

            if (this.wearsGlasses) {
                // Enhanced input for glasses mode: 21 features
                inputVector = [
                    features.leftIrisRatioX,
                    features.leftIrisRatioY,
                    features.rightIrisRatioX,
                    features.rightIrisRatioY,
                    features.avgIrisRatioX,
                    features.avgIrisRatioY,
                    features.headPitch,
                    features.headYaw,
                    features.headRoll,
                    features.leftAperture || 0,
                    features.rightAperture || 0,
                    features.avgAperture || 0,
                    features.leftIrisDepth || 0,
                    features.rightIrisDepth || 0,
                    features.depthDifference || 0,
                    features.leftEyeWidth || 0,
                    features.rightEyeWidth || 0,
                    features.leftVerticalPos || 0,
                    features.rightVerticalPos || 0,
                    sample.rawGazeX / window.innerWidth,
                    sample.rawGazeY / window.innerHeight
                ];
            } else {
                // Standard input: 12 features
                inputVector = [
                    features.leftIrisRatioX,
                    features.leftIrisRatioY,
                    features.rightIrisRatioX,
                    features.rightIrisRatioY,
                    features.avgIrisRatioX,
                    features.avgIrisRatioY,
                    features.headPitch,
                    features.headYaw,
                    features.headRoll,
                    features.avgAperture || 0,
                    sample.rawGazeX / window.innerWidth,
                    sample.rawGazeY / window.innerHeight
                ];
            }

            inputs.push(inputVector);

            // Output: target screen coordinates (normalized)
            outputs.push([
                sample.targetX / window.innerWidth,
                sample.targetY / window.innerHeight
            ]);
        }

        return { inputs, outputs };
    }

    computeNormalization(inputs) {
        const numFeatures = inputs[0].length;
        this.featureMean = new Array(numFeatures).fill(0);
        this.featureStd = new Array(numFeatures).fill(0);

        // Calculate mean
        for (const input of inputs) {
            for (let i = 0; i < numFeatures; i++) {
                this.featureMean[i] += input[i];
            }
        }
        for (let i = 0; i < numFeatures; i++) {
            this.featureMean[i] /= inputs.length;
        }

        // Calculate standard deviation
        for (const input of inputs) {
            for (let i = 0; i < numFeatures; i++) {
                this.featureStd[i] += Math.pow(input[i] - this.featureMean[i], 2);
            }
        }
        for (let i = 0; i < numFeatures; i++) {
            this.featureStd[i] = Math.sqrt(this.featureStd[i] / inputs.length);
            if (this.featureStd[i] === 0) this.featureStd[i] = 1; // Avoid division by zero
        }
    }

    normalizeFeatures(inputs) {
        return inputs.map(input =>
            input.map((val, i) => (val - this.featureMean[i]) / this.featureStd[i])
        );
    }

    correct(rawX, rawY, features) {
        if (!this.isModelTrained) {
            // If not trained, apply basic filtering only
            return this.filter.filter(rawX, rawY);
        }

        // Prepare input features based on mode
        let input;

        if (this.wearsGlasses) {
            input = [
                features.leftIrisRatioX,
                features.leftIrisRatioY,
                features.rightIrisRatioX,
                features.rightIrisRatioY,
                features.avgIrisRatioX,
                features.avgIrisRatioY,
                features.headPitch,
                features.headYaw,
                features.headRoll,
                features.leftAperture || 0,
                features.rightAperture || 0,
                features.avgAperture || 0,
                features.leftIrisDepth || 0,
                features.rightIrisDepth || 0,
                features.depthDifference || 0,
                features.leftEyeWidth || 0,
                features.rightEyeWidth || 0,
                features.leftVerticalPos || 0,
                features.rightVerticalPos || 0,
                rawX / window.innerWidth,
                rawY / window.innerHeight
            ];
        } else {
            input = [
                features.leftIrisRatioX,
                features.leftIrisRatioY,
                features.rightIrisRatioX,
                features.rightIrisRatioY,
                features.avgIrisRatioX,
                features.avgIrisRatioY,
                features.headPitch,
                features.headYaw,
                features.headRoll,
                features.avgAperture || 0,
                rawX / window.innerWidth,
                rawY / window.innerHeight
            ];
        }

        // Normalize input
        const normalizedInput = input.map((val, i) =>
            (val - this.featureMean[i]) / this.featureStd[i]
        );

        // Predict corrected gaze
        const inputTensor = tf.tensor2d([normalizedInput]);
        const prediction = this.model.predict(inputTensor);
        const [predX, predY] = prediction.dataSync();

        // Clean up tensors
        inputTensor.dispose();
        prediction.dispose();

        // Convert back to screen coordinates
        const correctedX = predX * window.innerWidth;
        const correctedY = predY * window.innerHeight;

        // Apply temporal filtering
        return this.filter.filter(correctedX, correctedY);
    }
}
