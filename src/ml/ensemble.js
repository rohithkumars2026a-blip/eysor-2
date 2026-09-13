import * as tf from '@tensorflow/tfjs';
import { GazeFilter } from '../core/filtering.js';

/**
 * Ensemble Multi-Model System for Ultra-High Precision Gaze Estimation
 * Target: < 0.5px error through model fusion and adaptive learning
 */
export class EnsembleCorrector {
    constructor(wearsGlasses = false) {
        this.wearsGlasses = wearsGlasses;
        this.models = {
            primary: null,      // Deep MLP - main predictor
            residual: null,     // Residual error correction network
            attention: null,    // Attention-based feature weighting
            temporal: null,     // LSTM for temporal consistency
            voting: null        // Meta-learner for ensemble fusion
        };

        // Adaptive filtering with dynamic parameters
        this.filter = new GazeFilter(
            wearsGlasses ? 0.5 : 0.8,
            wearsGlasses ? 0.015 : 0.01,
            wearsGlasses ? 0.6 : 0.8
        );

        this.isModelTrained = false;
        this.featureMean = null;
        this.featureStd = null;
        this.modelId = wearsGlasses ? 'gaze-ensemble-glasses-v2' : 'gaze-ensemble-v2';

        // Temporal buffer for LSTM
        this.temporalBuffer = [];
        this.temporalWindowSize = 10;

        // Online learning buffer
        this.onlineLearningBuffer = [];
        this.onlineLearningEnabled = true;
        this.retrainingThreshold = 100; // samples before retraining
    }

    async init() {
        console.log(`Initializing Ensemble Gaze Corrector (${this.wearsGlasses ? 'Glasses' : 'Standard'} mode)...`);

        try {
            // Try to load existing models
            await this.loadModels();
            console.log('Loaded existing ensemble models');
        } catch (error) {
            console.log('No existing models found, will create new ensemble');
            this.createModels();
        }
    }

    async loadModels() {
        const loadPromises = Object.keys(this.models).map(async (key) => {
            const modelPath = `indexeddb://${this.modelId}-${key}`;
            this.models[key] = await tf.loadLayersModel(modelPath);
        });

        await Promise.all(loadPromises);
        this.isModelTrained = true;
    }

    createModels() {
        const inputSize = this.wearsGlasses ? 21 : 12;

        // 1. Primary Deep MLP - Main predictor with skip connections
        this.models.primary = this.createPrimaryModel(inputSize);

        // 2. Residual Network - Learns to correct primary model errors
        this.models.residual = this.createResidualModel(inputSize);

        // 3. Attention Network - Learns feature importance dynamically
        this.models.attention = this.createAttentionModel(inputSize);

        // 4. Temporal LSTM - Smooths predictions over time
        this.models.temporal = this.createTemporalModel(inputSize);

        // 5. Voting Meta-Learner - Combines all model outputs
        this.models.voting = this.createVotingModel();

        console.log(`Created ensemble with ${Object.keys(this.models).length} models`);
    }

    createPrimaryModel(inputSize) {
        const units1 = this.wearsGlasses ? 256 : 128;
        const units2 = this.wearsGlasses ? 128 : 64;
        const units3 = this.wearsGlasses ? 64 : 32;

        const model = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [inputSize],
                    units: units1,
                    activation: 'relu',
                    kernelInitializer: 'heNormal',
                    kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
                }),
                tf.layers.batchNormalization(),
                tf.layers.dropout({ rate: 0.3 }),

                tf.layers.dense({
                    units: units2,
                    activation: 'relu',
                    kernelInitializer: 'heNormal'
                }),
                tf.layers.batchNormalization(),
                tf.layers.dropout({ rate: 0.25 }),

                tf.layers.dense({
                    units: units3,
                    activation: 'relu',
                    kernelInitializer: 'heNormal'
                }),
                tf.layers.dropout({ rate: 0.15 }),

                tf.layers.dense({
                    units: 2,
                    activation: 'linear'
                })
            ]
        });

        model.compile({
            optimizer: tf.train.adam(0.0005),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        return model;
    }

    createResidualModel(inputSize) {
        // Takes: original features + primary prediction
        const model = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [inputSize + 2], // features + primary output
                    units: 64,
                    activation: 'relu',
                    kernelInitializer: 'heNormal'
                }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({
                    units: 32,
                    activation: 'relu'
                }),
                tf.layers.dense({
                    units: 2,
                    activation: 'linear' // Residual correction
                })
            ]
        });

        model.compile({
            optimizer: tf.train.adam(0.0003),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        return model;
    }

    createAttentionModel(inputSize) {
        // Self-attention mechanism for feature weighting
        const model = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [inputSize],
                    units: 128,
                    activation: 'relu',
                    kernelInitializer: 'heNormal'
                }),
                tf.layers.dropout({ rate: 0.25 }),
                tf.layers.dense({
                    units: 64,
                    activation: 'relu'
                }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({
                    units: 32,
                    activation: 'relu'
                }),
                tf.layers.dense({
                    units: 2,
                    activation: 'linear'
                })
            ]
        });

        model.compile({
            optimizer: tf.train.adam(0.0004),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        return model;
    }

    createTemporalModel(inputSize) {
        // LSTM for temporal smoothing
        const model = tf.sequential({
            layers: [
                tf.layers.lstm({
                    inputShape: [this.temporalWindowSize, inputSize],
                    units: 64,
                    returnSequences: false
                }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({
                    units: 32,
                    activation: 'relu'
                }),
                tf.layers.dense({
                    units: 2,
                    activation: 'linear'
                })
            ]
        });

        model.compile({
            optimizer: tf.train.adam(0.0003),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        return model;
    }

    createVotingModel() {
        // Meta-learner: takes outputs from all 4 models and fuses them
        const model = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [8], // 4 models × 2 outputs each
                    units: 16,
                    activation: 'relu',
                    kernelInitializer: 'heNormal'
                }),
                tf.layers.dense({
                    units: 8,
                    activation: 'relu'
                }),
                tf.layers.dense({
                    units: 2,
                    activation: 'linear'
                })
            ]
        });

        model.compile({
            optimizer: tf.train.adam(0.0005),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        return model;
    }

    async train(calibrationData) {
        if (!calibrationData || calibrationData.length === 0) {
            throw new Error('No calibration data provided');
        }

        console.log(`Training ensemble on ${calibrationData.length} samples...`);

        const { inputs, outputs } = this.prepareTrainingData(calibrationData);
        this.computeNormalization(inputs);
        const normalizedInputs = this.normalizeFeatures(inputs);

        // Convert to tensors
        const xTensor = tf.tensor2d(normalizedInputs);
        const yTensor = tf.tensor2d(outputs);

        // Extended training epochs for ultra-precision
        const epochs = this.wearsGlasses ? 120 : 100;
        const batchSize = 8; // Smaller batches for better convergence

        // Train each model in sequence
        console.log('Training Primary Model...');
        await this.trainModel(this.models.primary, xTensor, yTensor, epochs, batchSize);

        // Generate primary predictions for residual training
        const primaryPreds = this.models.primary.predict(xTensor);
        const residuals = tf.sub(yTensor, primaryPreds);

        // Create residual training data (features + primary output → residual)
        const residualInputs = tf.concat([xTensor, primaryPreds], 1);

        console.log('Training Residual Model...');
        await this.trainModel(this.models.residual, residualInputs, residuals, epochs, batchSize);

        console.log('Training Attention Model...');
        await this.trainModel(this.models.attention, xTensor, yTensor, epochs, batchSize);

        // Prepare temporal sequences for LSTM
        console.log('Training Temporal Model...');
        const { temporalX, temporalY } = this.createTemporalSequences(normalizedInputs, outputs);
        if (temporalX.length > 0) {
            const temporalXTensor = tf.tensor3d(temporalX);
            const temporalYTensor = tf.tensor2d(temporalY);
            await this.trainModel(this.models.temporal, temporalXTensor, temporalYTensor, epochs, batchSize);
            temporalXTensor.dispose();
            temporalYTensor.dispose();
        }

        // Train voting meta-learner
        console.log('Training Voting Meta-Learner...');
        await this.trainVotingModel(xTensor, yTensor, epochs);

        // Cleanup
        xTensor.dispose();
        yTensor.dispose();
        primaryPreds.dispose();
        residuals.dispose();
        residualInputs.dispose();

        // Save all models
        await this.saveModels();

        this.isModelTrained = true;
        console.log('Ensemble training complete!');
    }

    async trainModel(model, xTensor, yTensor, epochs, batchSize) {
        await model.fit(xTensor, yTensor, {
            epochs: epochs,
            batchSize: batchSize,
            validationSplit: 0.15,
            shuffle: true,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    if (epoch % 20 === 0) {
                        console.log(`  Epoch ${epoch}: loss=${logs.loss.toFixed(5)}, mae=${logs.mae.toFixed(3)}px`);
                    }
                }
            }
        });
    }

    async trainVotingModel(xTensor, yTensor, epochs) {
        // Get predictions from all 4 models
        const primaryPred = this.models.primary.predict(xTensor);
        const attentionPred = this.models.attention.predict(xTensor);

        // For residual: concat features + primary prediction
        const residualInput = tf.concat([xTensor, primaryPred], 1);
        const residualPred = this.models.residual.predict(residualInput);
        const correctedPrimary = tf.add(primaryPred, residualPred);

        // For temporal: use last prediction (since we can't create full sequences here)
        const temporalPred = primaryPred.clone(); // Fallback

        // Concatenate all predictions
        const votingInput = tf.concat([primaryPred, correctedPrimary, attentionPred, temporalPred], 1);

        await this.models.voting.fit(votingInput, yTensor, {
            epochs: epochs,
            batchSize: 8,
            validationSplit: 0.15,
            shuffle: true
        });

        // Cleanup
        primaryPred.dispose();
        attentionPred.dispose();
        residualInput.dispose();
        residualPred.dispose();
        correctedPrimary.dispose();
        temporalPred.dispose();
        votingInput.dispose();
    }

    createTemporalSequences(inputs, outputs) {
        const sequences = [];
        const labels = [];

        for (let i = this.temporalWindowSize; i < inputs.length; i++) {
            const sequence = inputs.slice(i - this.temporalWindowSize, i);
            sequences.push(sequence);
            labels.push(outputs[i]);
        }

        return { temporalX: sequences, temporalY: labels };
    }

    prepareTrainingData(calibrationData) {
        const inputs = [];
        const outputs = [];

        for (const sample of calibrationData) {
            const features = sample.features;
            let inputVector;

            if (this.wearsGlasses) {
                inputVector = [
                    features.leftIrisRatioX, features.leftIrisRatioY,
                    features.rightIrisRatioX, features.rightIrisRatioY,
                    features.avgIrisRatioX, features.avgIrisRatioY,
                    features.headPitch, features.headYaw, features.headRoll,
                    features.leftAperture || 0, features.rightAperture || 0, features.avgAperture || 0,
                    features.leftIrisDepth || 0, features.rightIrisDepth || 0, features.depthDifference || 0,
                    features.leftEyeWidth || 0, features.rightEyeWidth || 0,
                    features.leftVerticalPos || 0, features.rightVerticalPos || 0,
                    sample.rawGazeX / window.innerWidth, sample.rawGazeY / window.innerHeight
                ];
            } else {
                inputVector = [
                    features.leftIrisRatioX, features.leftIrisRatioY,
                    features.rightIrisRatioX, features.rightIrisRatioY,
                    features.avgIrisRatioX, features.avgIrisRatioY,
                    features.headPitch, features.headYaw, features.headRoll,
                    features.avgAperture || 0,
                    sample.rawGazeX / window.innerWidth, sample.rawGazeY / window.innerHeight
                ];
            }

            inputs.push(inputVector);
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

        for (const input of inputs) {
            for (let i = 0; i < numFeatures; i++) {
                this.featureMean[i] += input[i];
            }
        }
        for (let i = 0; i < numFeatures; i++) {
            this.featureMean[i] /= inputs.length;
        }

        for (const input of inputs) {
            for (let i = 0; i < numFeatures; i++) {
                this.featureStd[i] += Math.pow(input[i] - this.featureMean[i], 2);
            }
        }
        for (let i = 0; i < numFeatures; i++) {
            this.featureStd[i] = Math.sqrt(this.featureStd[i] / inputs.length);
            if (this.featureStd[i] === 0) this.featureStd[i] = 1;
        }
    }

    normalizeFeatures(inputs) {
        return inputs.map(input =>
            input.map((val, i) => (val - this.featureMean[i]) / this.featureStd[i])
        );
    }

    correct(rawX, rawY, features) {
        if (!this.isModelTrained) {
            return this.filter.filter(rawX, rawY);
        }

        // Prepare input
        let input;
        if (this.wearsGlasses) {
            input = [
                features.leftIrisRatioX, features.leftIrisRatioY,
                features.rightIrisRatioX, features.rightIrisRatioY,
                features.avgIrisRatioX, features.avgIrisRatioY,
                features.headPitch, features.headYaw, features.headRoll,
                features.leftAperture || 0, features.rightAperture || 0, features.avgAperture || 0,
                features.leftIrisDepth || 0, features.rightIrisDepth || 0, features.depthDifference || 0,
                features.leftEyeWidth || 0, features.rightEyeWidth || 0,
                features.leftVerticalPos || 0, features.rightVerticalPos || 0,
                rawX / window.innerWidth, rawY / window.innerHeight
            ];
        } else {
            input = [
                features.leftIrisRatioX, features.leftIrisRatioY,
                features.rightIrisRatioX, features.rightIrisRatioY,
                features.avgIrisRatioX, features.avgIrisRatioY,
                features.headPitch, features.headYaw, features.headRoll,
                features.avgAperture || 0,
                rawX / window.innerWidth, rawY / window.innerHeight
            ];
        }

        const normalizedInput = input.map((val, i) =>
            (val - this.featureMean[i]) / this.featureStd[i]
        );

        // Get ensemble predictions
        const inputTensor = tf.tensor2d([normalizedInput]);

        // Primary prediction
        const primaryPred = this.models.primary.predict(inputTensor);

        // Residual correction
        const residualInput = tf.concat([inputTensor, primaryPred], 1);
        const residualPred = this.models.residual.predict(residualInput);
        const correctedPrimary = tf.add(primaryPred, residualPred);

        // Attention prediction
        const attentionPred = this.models.attention.predict(inputTensor);

        // Temporal prediction (with buffer)
        let temporalPred;
        this.temporalBuffer.push(normalizedInput);
        if (this.temporalBuffer.length > this.temporalWindowSize) {
            this.temporalBuffer.shift();
        }

        if (this.temporalBuffer.length === this.temporalWindowSize) {
            const temporalInput = tf.tensor3d([this.temporalBuffer]);
            temporalPred = this.models.temporal.predict(temporalInput);
            temporalInput.dispose();
        } else {
            temporalPred = primaryPred.clone();
        }

        // Voting ensemble
        const votingInput = tf.concat([primaryPred, correctedPrimary, attentionPred, temporalPred], 1);
        const finalPred = this.models.voting.predict(votingInput);

        const [predX, predY] = finalPred.dataSync();

        // Cleanup
        inputTensor.dispose();
        primaryPred.dispose();
        residualInput.dispose();
        residualPred.dispose();
        correctedPrimary.dispose();
        attentionPred.dispose();
        temporalPred.dispose();
        votingInput.dispose();
        finalPred.dispose();

        // Convert back to screen coordinates
        const correctedX = predX * window.innerWidth;
        const correctedY = predY * window.innerHeight;

        // Apply adaptive filtering
        return this.filter.filter(correctedX, correctedY);
    }

    async saveModels() {
        console.log('Saving ensemble models...');
        const savePromises = Object.entries(this.models).map(async ([key, model]) => {
            const modelPath = `indexeddb://${this.modelId}-${key}`;
            await model.save(modelPath);
        });

        await Promise.all(savePromises);
        console.log('All models saved successfully');
    }
}
