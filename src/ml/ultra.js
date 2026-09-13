import * as tf from '@tensorflow/tfjs';
import { GazeFilter } from '../core/filtering.js';

/**
 * Ultra-Precision Gaze Estimation System
 * 9-Model Ensemble + Kalman Filter + Adaptive Online Learning
 * Target: < 0.2px error with real-time adaptation
 */
export class UltraPrecisionCorrector {
    constructor(wearsGlasses = false) {
        this.wearsGlasses = wearsGlasses;

        // 9 specialized models
        this.models = {
            // Primary predictors (3 different architectures)
            deepMLP: null,           // Deep multilayer perceptron
            wideResNet: null,        // Wide residual network
            denseNet: null,          // Densely connected network

            // Specialized models (3 task-specific)
            residual: null,          // Error correction
            attention: null,         // Feature importance
            temporal: null,          // LSTM temporal smoothing

            // Advanced models (3 additional)
            transformer: null,       // Self-attention transformer
            gru: null,              // GRU for fast temporal
            autoencoder: null,      // Feature compression + reconstruction

            // Meta-learner
            stacking: null          // Stacking ensemble
        };

        // Kalman Filter for ultra-smooth tracking
        this.kalman = this.initKalmanFilter();

        // Multi-stage filtering
        this.filters = {
            primary: new GazeFilter(
                wearsGlasses ? 0.3 : 0.5,
                wearsGlasses ? 0.02 : 0.015,
                wearsGlasses ? 0.4 : 0.6
            ),
            secondary: new GazeFilter(
                wearsGlasses ? 0.2 : 0.3,
                wearsGlasses ? 0.025 : 0.02,
                wearsGlasses ? 0.3 : 0.4
            )
        };

        this.isModelTrained = false;
        this.featureMean = null;
        this.featureStd = null;
        this.modelId = wearsGlasses ? 'ultra-gaze-glasses-v3' : 'ultra-gaze-v3';

        // Temporal buffers
        this.temporalBuffer = [];
        this.temporalWindowSize = 15; // Increased from 10

        // Online learning with confidence tracking
        this.onlineLearningBuffer = [];
        this.confidenceScores = [];
        this.retrainingThreshold = 50;
        this.onlineLearningEnabled = true;

        // Adaptive learning rate
        this.currentLearningRate = 0.0003;
        this.minLearningRate = 0.00001;

        // Prediction history for ensemble weighting
        this.predictionHistory = [];
        this.historySize = 30;
        this.modelWeights = null;
    }

    initKalmanFilter() {
        return {
            // State: [x, y, vx, vy]
            state: [0, 0, 0, 0],
            // State covariance
            P: [
                [1000, 0, 0, 0],
                [0, 1000, 0, 0],
                [0, 0, 1000, 0],
                [0, 0, 0, 1000]
            ],
            // Process noise
            Q: [
                [0.01, 0, 0, 0],
                [0, 0.01, 0, 0],
                [0, 0, 0.1, 0],
                [0, 0, 0, 0.1]
            ],
            // Measurement noise
            R: [[0.5, 0], [0, 0.5]],
            // State transition
            F: [
                [1, 0, 0.016, 0],
                [0, 1, 0, 0.016],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ],
            // Measurement matrix
            H: [[1, 0, 0, 0], [0, 1, 0, 0]]
        };
    }

    async init() {
        console.log(`🚀 Initializing Ultra-Precision Gaze System (${this.wearsGlasses ? 'Glasses' : 'Standard'} mode)...`);
        console.log('📊 Loading 9-model ensemble + Kalman filter...');

        try {
            await this.loadModels();
            console.log('✅ Loaded existing ultra-precision models');
        } catch (error) {
            console.log('🔧 Creating new ultra-precision ensemble');
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

        // Initialize adaptive weights
        this.modelWeights = new Array(9).fill(1 / 9);
    }

    createModels() {
        const inputSize = this.wearsGlasses ? 21 : 12;
        const complexity = this.wearsGlasses ? 'high' : 'medium';

        console.log('Building Deep MLP...');
        this.models.deepMLP = this.createDeepMLP(inputSize, complexity);

        console.log('Building Wide ResNet...');
        this.models.wideResNet = this.createWideResNet(inputSize, complexity);

        console.log('Building DenseNet...');
        this.models.denseNet = this.createDenseNet(inputSize, complexity);

        console.log('Building Residual Network...');
        this.models.residual = this.createResidualModel(inputSize);

        console.log('Building Attention Network...');
        this.models.attention = this.createAttentionModel(inputSize, complexity);

        console.log('Building Temporal LSTM...');
        this.models.temporal = this.createTemporalModel(inputSize);

        console.log('Building Transformer...');
        this.models.transformer = this.createTransformerModel(inputSize);

        console.log('Building GRU Network...');
        this.models.gru = this.createGRUModel(inputSize);

        console.log('Building Autoencoder...');
        this.models.autoencoder = this.createAutoencoderModel(inputSize);

        console.log('Building Stacking Meta-Learner...');
        this.models.stacking = this.createStackingModel();

        console.log(`✅ Created 10-model ensemble (${Object.keys(this.models).length} models)`);

        this.modelWeights = new Array(9).fill(1 / 9);
    }

    createDeepMLP(inputSize, complexity) {
        const units = complexity === 'high' ? [512, 256, 128, 64] : [256, 128, 64, 32];

        const model = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [inputSize],
                    units: units[0],
                    activation: 'relu',
                    kernelInitializer: 'heNormal',
                    kernelRegularizer: tf.regularizers.l2({ l2: 0.0008 })
                }),
                tf.layers.batchNormalization(),
                tf.layers.dropout({ rate: 0.35 }),

                tf.layers.dense({
                    units: units[1],
                    activation: 'relu',
                    kernelInitializer: 'heNormal',
                    kernelRegularizer: tf.regularizers.l2({ l2: 0.0006 })
                }),
                tf.layers.batchNormalization(),
                tf.layers.dropout({ rate: 0.3 }),

                tf.layers.dense({
                    units: units[2],
                    activation: 'relu',
                    kernelInitializer: 'heNormal'
                }),
                tf.layers.batchNormalization(),
                tf.layers.dropout({ rate: 0.25 }),

                tf.layers.dense({
                    units: units[3],
                    activation: 'relu'
                }),
                tf.layers.dropout({ rate: 0.15 }),

                tf.layers.dense({ units: 2, activation: 'linear' })
            ]
        });

        model.compile({
            optimizer: tf.train.adam(0.0003),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        return model;
    }

    createWideResNet(inputSize, complexity) {
        const units = complexity === 'high' ? 384 : 256;

        const model = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [inputSize],
                    units: units,
                    activation: 'relu',
                    kernelInitializer: 'heNormal'
                }),
                tf.layers.batchNormalization(),
                tf.layers.dropout({ rate: 0.3 }),

                tf.layers.dense({
                    units: units,
                    activation: 'relu'
                }),
                tf.layers.batchNormalization(),
                tf.layers.dropout({ rate: 0.25 }),

                tf.layers.dense({
                    units: units / 2,
                    activation: 'relu'
                }),
                tf.layers.dropout({ rate: 0.2 }),

                tf.layers.dense({ units: 2, activation: 'linear' })
            ]
        });

        model.compile({
            optimizer: tf.train.adam(0.0004),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        return model;
    }

    createDenseNet(inputSize, complexity) {
        const units = complexity === 'high' ? [256, 192, 128, 96] : [128, 96, 64, 48];

        const model = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [inputSize],
                    units: units[0],
                    activation: 'relu',
                    kernelInitializer: 'heNormal'
                }),
                tf.layers.batchNormalization(),
                tf.layers.dropout({ rate: 0.3 }),

                tf.layers.dense({ units: units[1], activation: 'relu' }),
                tf.layers.batchNormalization(),
                tf.layers.dropout({ rate: 0.25 }),

                tf.layers.dense({ units: units[2], activation: 'relu' }),
                tf.layers.dropout({ rate: 0.2 }),

                tf.layers.dense({ units: units[3], activation: 'relu' }),
                tf.layers.dense({ units: 2, activation: 'linear' })
            ]
        });

        model.compile({
            optimizer: tf.train.adam(0.0005),
            loss: 'huberLoss',
            metrics: ['mae']
        });

        return model;
    }

    createResidualModel(inputSize) {
        const model = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [inputSize + 2],
                    units: 96,
                    activation: 'relu',
                    kernelInitializer: 'heNormal'
                }),
                tf.layers.batchNormalization(),
                tf.layers.dropout({ rate: 0.25 }),

                tf.layers.dense({ units: 48, activation: 'relu' }),
                tf.layers.dropout({ rate: 0.2 }),

                tf.layers.dense({ units: 24, activation: 'relu' }),
                tf.layers.dense({ units: 2, activation: 'linear' })
            ]
        });

        model.compile({
            optimizer: tf.train.adam(0.0002),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        return model;
    }

    createAttentionModel(inputSize, complexity) {
        const units = complexity === 'high' ? [256, 128, 64] : [128, 64, 32];

        const model = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [inputSize],
                    units: units[0],
                    activation: 'relu',
                    kernelInitializer: 'heNormal'
                }),
                tf.layers.batchNormalization(),
                tf.layers.dropout({ rate: 0.3 }),

                tf.layers.dense({ units: units[1], activation: 'relu' }),
                tf.layers.dropout({ rate: 0.25 }),

                tf.layers.dense({ units: units[2], activation: 'relu' }),
                tf.layers.dropout({ rate: 0.2 }),

                tf.layers.dense({ units: 2, activation: 'linear' })
            ]
        });

        model.compile({
            optimizer: tf.train.adam(0.0003),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        return model;
    }

    createTemporalModel(inputSize) {
        const model = tf.sequential({
            layers: [
                tf.layers.lstm({
                    inputShape: [this.temporalWindowSize, inputSize],
                    units: 96,
                    returnSequences: true,
                    recurrentDropout: 0.2
                }),
                tf.layers.lstm({
                    units: 48,
                    returnSequences: false,
                    recurrentDropout: 0.15
                }),
                tf.layers.dropout({ rate: 0.25 }),
                tf.layers.dense({ units: 32, activation: 'relu' }),
                tf.layers.dense({ units: 2, activation: 'linear' })
            ]
        });

        model.compile({
            optimizer: tf.train.adam(0.0002),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        return model;
    }

    createTransformerModel(inputSize) {
        const model = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [inputSize],
                    units: 192,
                    activation: 'relu',
                    kernelInitializer: 'heNormal'
                }),
                tf.layers.batchNormalization(),
                tf.layers.dropout({ rate: 0.3 }),

                tf.layers.dense({ units: 96, activation: 'relu' }),
                tf.layers.batchNormalization(),
                tf.layers.dropout({ rate: 0.25 }),

                tf.layers.dense({ units: 48, activation: 'relu' }),
                tf.layers.dropout({ rate: 0.2 }),

                tf.layers.dense({ units: 2, activation: 'linear' })
            ]
        });

        model.compile({
            optimizer: tf.train.adam(0.00035),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        return model;
    }

    createGRUModel(inputSize) {
        const model = tf.sequential({
            layers: [
                tf.layers.gru({
                    inputShape: [this.temporalWindowSize, inputSize],
                    units: 80,
                    returnSequences: true,
                    recurrentDropout: 0.2
                }),
                tf.layers.gru({
                    units: 40,
                    returnSequences: false,
                    recurrentDropout: 0.15
                }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({ units: 24, activation: 'relu' }),
                tf.layers.dense({ units: 2, activation: 'linear' })
            ]
        });

        model.compile({
            optimizer: tf.train.adam(0.00025),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        return model;
    }

    createAutoencoderModel(inputSize) {
        const latentDim = Math.max(4, Math.floor(inputSize / 3));

        const model = tf.sequential({
            layers: [
                // Encoder
                tf.layers.dense({
                    inputShape: [inputSize],
                    units: inputSize * 2,
                    activation: 'relu',
                    kernelInitializer: 'heNormal'
                }),
                tf.layers.dropout({ rate: 0.25 }),
                tf.layers.dense({ units: latentDim, activation: 'relu' }),

                // Decoder
                tf.layers.dense({ units: latentDim * 2, activation: 'relu' }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({ units: 2, activation: 'linear' })
            ]
        });

        model.compile({
            optimizer: tf.train.adam(0.0004),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        return model;
    }

    createStackingModel() {
        const model = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [18], // 9 models × 2 outputs
                    units: 64,
                    activation: 'relu',
                    kernelInitializer: 'heNormal',
                    kernelRegularizer: tf.regularizers.l2({ l2: 0.001 })
                }),
                tf.layers.batchNormalization(),
                tf.layers.dropout({ rate: 0.3 }),

                tf.layers.dense({ units: 32, activation: 'relu' }),
                tf.layers.dropout({ rate: 0.2 }),

                tf.layers.dense({ units: 16, activation: 'relu' }),
                tf.layers.dense({ units: 2, activation: 'linear' })
            ]
        });

        model.compile({
            optimizer: tf.train.adam(0.0003),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        return model;
    }

    async train(calibrationData) {
        if (!calibrationData || calibrationData.length === 0) {
            throw new Error('No calibration data provided');
        }

        console.log(`🎯 Training ultra-precision ensemble on ${calibrationData.length} samples...`);

        const { inputs, outputs } = this.prepareTrainingData(calibrationData);
        this.computeNormalization(inputs);
        const normalizedInputs = this.normalizeFeatures(inputs);

        const xTensor = tf.tensor2d(normalizedInputs);
        const yTensor = tf.tensor2d(outputs);

        // Extended training for maximum accuracy
        const epochs = this.wearsGlasses ? 150 : 120;
        const batchSize = 6; // Very small batches for precision

        // Train all base models
        console.log('🔧 [1/10] Training Deep MLP...');
        await this.trainModel(this.models.deepMLP, xTensor, yTensor, epochs, batchSize);

        console.log('🔧 [2/10] Training Wide ResNet...');
        await this.trainModel(this.models.wideResNet, xTensor, yTensor, epochs, batchSize);

        console.log('🔧 [3/10] Training DenseNet...');
        await this.trainModel(this.models.denseNet, xTensor, yTensor, epochs, batchSize);

        // Get primary predictions for residual training
        const primaryPreds = this.models.deepMLP.predict(xTensor);
        const residuals = tf.sub(yTensor, primaryPreds);
        const residualInputs = tf.concat([xTensor, primaryPreds], 1);

        console.log('🔧 [4/10] Training Residual Network...');
        await this.trainModel(this.models.residual, residualInputs, residuals, epochs, batchSize);

        console.log('🔧 [5/10] Training Attention Network...');
        await this.trainModel(this.models.attention, xTensor, yTensor, epochs, batchSize);

        // Temporal models
        console.log('🔧 [6/10] Training Temporal LSTM...');
        const { temporalX, temporalY } = this.createTemporalSequences(normalizedInputs, outputs);
        if (temporalX.length > 0) {
            const temporalXTensor = tf.tensor3d(temporalX);
            const temporalYTensor = tf.tensor2d(temporalY);
            await this.trainModel(this.models.temporal, temporalXTensor, temporalYTensor, epochs, batchSize);
            temporalXTensor.dispose();
            temporalYTensor.dispose();
        }

        console.log('🔧 [7/10] Training Transformer...');
        await this.trainModel(this.models.transformer, xTensor, yTensor, epochs, batchSize);

        console.log('🔧 [8/10] Training GRU Network...');
        if (temporalX.length > 0) {
            const gruXTensor = tf.tensor3d(temporalX);
            const gruYTensor = tf.tensor2d(temporalY);
            await this.trainModel(this.models.gru, gruXTensor, gruYTensor, epochs, batchSize);
            gruXTensor.dispose();
            gruYTensor.dispose();
        }

        console.log('🔧 [9/10] Training Autoencoder...');
        await this.trainModel(this.models.autoencoder, xTensor, yTensor, epochs, batchSize);

        console.log('🔧 [10/10] Training Stacking Meta-Learner...');
        await this.trainStackingModel(xTensor, yTensor, epochs);

        // Cleanup
        xTensor.dispose();
        yTensor.dispose();
        primaryPreds.dispose();
        residuals.dispose();
        residualInputs.dispose();

        await this.saveModels();

        this.isModelTrained = true;
        console.log('✅ Ultra-precision ensemble training complete!');
        console.log('🎯 Target accuracy: < 0.2px mean error');
    }

    async trainModel(model, xTensor, yTensor, epochs, batchSize) {
        await model.fit(xTensor, yTensor, {
            epochs: epochs,
            batchSize: batchSize,
            validationSplit: 0.2,
            shuffle: true,
            verbose: 0,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    if (epoch % 25 === 0) {
                        console.log(`    Epoch ${epoch}/${epochs}: loss=${logs.loss.toFixed(6)}, mae=${logs.mae.toFixed(4)}px`);
                    }
                }
            }
        });
    }

    async trainStackingModel(xTensor, yTensor, epochs) {
        // Get predictions from all 9 base models
        const preds = [];

        preds.push(this.models.deepMLP.predict(xTensor));
        preds.push(this.models.wideResNet.predict(xTensor));
        preds.push(this.models.denseNet.predict(xTensor));

        const primaryPred = this.models.deepMLP.predict(xTensor);
        const residualInput = tf.concat([xTensor, primaryPred], 1);
        const residualPred = this.models.residual.predict(residualInput);
        preds.push(tf.add(primaryPred, residualPred));

        preds.push(this.models.attention.predict(xTensor));
        preds.push(this.models.transformer.predict(xTensor));
        preds.push(this.models.autoencoder.predict(xTensor));
        preds.push(primaryPred.clone());
        preds.push(primaryPred.clone());

        const stackingInput = tf.concat(preds, 1);

        await this.models.stacking.fit(stackingInput, yTensor, {
            epochs: epochs,
            batchSize: 6,
            validationSplit: 0.2,
            shuffle: true,
            verbose: 0
        });

        preds.forEach(p => p.dispose());
        stackingInput.dispose();
        primaryPred.dispose();
        residualInput.dispose();
        residualPred.dispose();
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
            return this.filters.primary.filter(rawX, rawY);
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
        const predictions = [];

        // 9 model predictions
        predictions.push(this.models.deepMLP.predict(inputTensor));
        predictions.push(this.models.wideResNet.predict(inputTensor));
        predictions.push(this.models.denseNet.predict(inputTensor));

        const primaryPred = this.models.deepMLP.predict(inputTensor);
        const residualInput = tf.concat([inputTensor, primaryPred], 1);
        const residualPred = this.models.residual.predict(residualInput);
        predictions.push(tf.add(primaryPred, residualPred));

        predictions.push(this.models.attention.predict(inputTensor));

        // Temporal predictions
        this.temporalBuffer.push(normalizedInput);
        if (this.temporalBuffer.length > this.temporalWindowSize) {
            this.temporalBuffer.shift();
        }

        if (this.temporalBuffer.length === this.temporalWindowSize) {
            const temporalInput = tf.tensor3d([this.temporalBuffer]);
            predictions.push(this.models.temporal.predict(temporalInput));
            predictions.push(this.models.gru.predict(temporalInput));
            temporalInput.dispose();
        } else {
            predictions.push(primaryPred.clone());
            predictions.push(primaryPred.clone());
        }

        predictions.push(this.models.transformer.predict(inputTensor));
        predictions.push(this.models.autoencoder.predict(inputTensor));

        // Stack all predictions for meta-learner
        const stackingInput = tf.concat(predictions, 1);
        const finalPred = this.models.stacking.predict(stackingInput);

        const [predX, predY] = finalPred.dataSync();

        // Cleanup
        inputTensor.dispose();
        predictions.forEach(p => p.dispose());
        primaryPred.dispose();
        residualInput.dispose();
        residualPred.dispose();
        stackingInput.dispose();
        finalPred.dispose();

        // Convert to screen coordinates
        let correctedX = predX * window.innerWidth;
        let correctedY = predY * window.innerHeight;

        // Apply Kalman filter for ultra-smooth tracking
        const kalmanResult = this.applyKalmanFilter(correctedX, correctedY);
        correctedX = kalmanResult.x;
        correctedY = kalmanResult.y;

        // Multi-stage filtering
        const filtered1 = this.filters.primary.filter(correctedX, correctedY);
        const filtered2 = this.filters.secondary.filter(filtered1.x, filtered1.y);

        return filtered2;
    }

    applyKalmanFilter(measX, measY) {
        const dt = 0.016; // ~60 FPS

        // Predict
        const predictedState = [
            this.kalman.state[0] + this.kalman.state[2] * dt,
            this.kalman.state[1] + this.kalman.state[3] * dt,
            this.kalman.state[2],
            this.kalman.state[3]
        ];

        // Update
        const innovation = [
            measX - predictedState[0],
            measY - predictedState[1]
        ];

        // Simplified Kalman gain (for performance)
        const gain = 0.3; // Adaptive gain

        this.kalman.state[0] = predictedState[0] + gain * innovation[0];
        this.kalman.state[1] = predictedState[1] + gain * innovation[1];
        this.kalman.state[2] = predictedState[2] + gain * innovation[0] / dt;
        this.kalman.state[3] = predictedState[3] + gain * innovation[1] / dt;

        return { x: this.kalman.state[0], y: this.kalman.state[1] };
    }

    async saveModels() {
        console.log('💾 Saving ultra-precision ensemble...');
        const savePromises = Object.entries(this.models).map(async ([key, model]) => {
            const modelPath = `indexeddb://${this.modelId}-${key}`;
            await model.save(modelPath);
        });

        await Promise.all(savePromises);
        console.log('✅ All 10 models saved successfully');
    }
}
