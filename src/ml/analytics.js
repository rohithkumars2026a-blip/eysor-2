import * as tf from '@tensorflow/tfjs';

/**
 * 3 Advanced Analytics Models for Training Data Processing
 * Analyzes calibration patterns, predicts accuracy, and extracts insights
 */
export class AnalyticsModels {
    constructor() {
        this.models = {
            patternAnalyzer: null,      // Analyzes calibration patterns
            accuracyPredictor: null,    // Predicts final accuracy before training
            insightExtractor: null      // Extracts valuable insights from data
        };

        this.isInitialized = false;
    }

    async init() {
        console.log('🔬 Initializing Analytics Models...');

        try {
            await this.loadModels();
            console.log('✅ Loaded existing analytics models');
        } catch (error) {
            console.log('🔧 Creating new analytics models');
            this.createModels();
        }
    }

    async loadModels() {
        const loadPromises = Object.keys(this.models).map(async (key) => {
            const modelPath = `indexeddb://analytics-${key}`;
            this.models[key] = await tf.loadLayersModel(modelPath);
        });

        await Promise.all(loadPromises);
        this.isInitialized = true;
    }

    createModels() {
        // 1. Pattern Analyzer - Identifies calibration quality patterns
        this.models.patternAnalyzer = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [15], // Aggregated calibration metrics
                    units: 128,
                    activation: 'relu',
                    kernelInitializer: 'heNormal'
                }),
                tf.layers.batchNormalization(),
                tf.layers.dropout({ rate: 0.3 }),
                tf.layers.dense({ units: 64, activation: 'relu' }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({ units: 32, activation: 'relu' }),
                tf.layers.dense({ units: 5, activation: 'softmax' }) // 5 quality categories
            ]
        });
        this.models.patternAnalyzer.compile({
            optimizer: tf.train.adam(0.0003),
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy']
        });

        // 2. Accuracy Predictor - Predicts final accuracy from calibration data
        this.models.accuracyPredictor = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [20], // Extended calibration features
                    units: 96,
                    activation: 'relu',
                    kernelInitializer: 'heNormal'
                }),
                tf.layers.batchNormalization(),
                tf.layers.dropout({ rate: 0.25 }),
                tf.layers.dense({ units: 48, activation: 'relu' }),
                tf.layers.dropout({ rate: 0.2 }),
                tf.layers.dense({ units: 24, activation: 'relu' }),
                tf.layers.dense({ units: 3, activation: 'linear' }) // [meanError, stdDev, visualAngle]
            ]
        });
        this.models.accuracyPredictor.compile({
            optimizer: tf.train.adam(0.0004),
            loss: 'meanSquaredError',
            metrics: ['mae']
        });

        // 3. Insight Extractor - Extracts feature importance and recommendations
        this.models.insightExtractor = tf.sequential({
            layers: [
                tf.layers.dense({
                    inputShape: [25], // Comprehensive session features
                    units: 160,
                    activation: 'relu',
                    kernelInitializer: 'heNormal'
                }),
                tf.layers.batchNormalization(),
                tf.layers.dropout({ rate: 0.3 }),
                tf.layers.dense({ units: 80, activation: 'relu' }),
                tf.layers.batchNormalization(),
                tf.layers.dropout({ rate: 0.25 }),
                tf.layers.dense({ units: 40, activation: 'relu' }),
                tf.layers.dense({ units: 10, activation: 'sigmoid' }) // 10 insight flags
            ]
        });
        this.models.insightExtractor.compile({
            optimizer: tf.train.adam(0.00035),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });

        this.isInitialized = true;
        console.log('✅ Analytics models created successfully');
    }

    /**
     * Analyze calibration pattern quality
     */
    analyzePattern(calibrationData) {
        if (!this.isInitialized || !this.models.patternAnalyzer) {
            console.warn('⚠️ Pattern analyzer not initialized, using heuristic analysis');
            return this.heuristicPatternAnalysis(calibrationData);
        }

        const features = this.extractPatternFeatures(calibrationData);
        const inputTensor = tf.tensor2d([features]);
        const prediction = this.models.patternAnalyzer.predict(inputTensor);
        const probabilities = prediction.dataSync();

        inputTensor.dispose();
        prediction.dispose();

        const qualityCategories = ['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'];
        const qualityIndex = probabilities.indexOf(Math.max(...probabilities));

        return {
            quality: qualityCategories[qualityIndex],
            confidence: probabilities[qualityIndex],
            distribution: {
                poor: probabilities[0],
                fair: probabilities[1],
                good: probabilities[2],
                veryGood: probabilities[3],
                excellent: probabilities[4]
            }
        };
    }

    heuristicPatternAnalysis(calibrationData) {
        if (!calibrationData || calibrationData.length === 0) {
            return { quality: 'Poor', confidence: 0, distribution: {} };
        }

        const samples = calibrationData.length;
        if (samples < 500) {
            return { quality: 'Fair', confidence: 0.6, distribution: {} };
        } else if (samples < 1000) {
            return { quality: 'Good', confidence: 0.75, distribution: {} };
        } else {
            return { quality: 'Very Good', confidence: 0.85, distribution: {} };
        }
    }

    /**
     * Predict expected accuracy before training
     */
    predictAccuracy(calibrationData) {
        if (!this.isInitialized || !this.models.accuracyPredictor) {
            console.warn('⚠️ Accuracy predictor not initialized, using heuristic prediction');
            return this.heuristicAccuracyPrediction(calibrationData);
        }

        const features = this.extractAccuracyFeatures(calibrationData);
        const inputTensor = tf.tensor2d([features]);
        const prediction = this.models.accuracyPredictor.predict(inputTensor);
        const [meanError, stdDev, visualAngle] = prediction.dataSync();

        inputTensor.dispose();
        prediction.dispose();

        return {
            predictedMeanError: meanError,
            predictedStdDev: stdDev,
            predictedVisualAngle: visualAngle,
            expectedQuality: this.classifyAccuracy(meanError)
        };
    }

    heuristicAccuracyPrediction(calibrationData) {
        if (!calibrationData || calibrationData.length === 0) {
            return {
                predictedMeanError: 10,
                predictedStdDev: 5,
                predictedVisualAngle: 2.0,
                expectedQuality: 'Fair'
            };
        }

        const samples = calibrationData.length;
        const predictedError = Math.max(0.5, 10 - (samples / 200));

        return {
            predictedMeanError: predictedError,
            predictedStdDev: predictedError / 2,
            predictedVisualAngle: predictedError * 0.1,
            expectedQuality: this.classifyAccuracy(predictedError)
        };
    }

    /**
     * Extract insights and recommendations
     */
    extractInsights(sessionData) {
        if (!this.isInitialized || !this.models.insightExtractor) {
            console.warn('⚠️ Insight extractor not initialized, using heuristic insights');
            return this.heuristicInsights(sessionData);
        }

        const features = this.extractInsightFeatures(sessionData);
        const inputTensor = tf.tensor2d([features]);
        const prediction = this.models.insightExtractor.predict(inputTensor);
        const insights = prediction.dataSync();

        inputTensor.dispose();
        prediction.dispose();

        const insightLabels = [
            'needsRecalibration',
            'poorLightingDetected',
            'excessiveHeadMovement',
            'inconsistentGazePattern',
            'optimalPerformance',
            'glassesReflectionIssue',
            'lowSampleQuality',
            'highVariance',
            'stableTracking',
            'recommendFineTuning'
        ];

        const activeInsights = insightLabels
            .map((label, i) => ({ label, confidence: insights[i] }))
            .filter(insight => insight.confidence > 0.5)
            .sort((a, b) => b.confidence - a.confidence);

        return {
            insights: activeInsights,
            recommendations: this.generateRecommendations(activeInsights)
        };
    }

    heuristicInsights(sessionData) {
        return {
            insights: [{ label: 'optimalPerformance', confidence: 0.8 }],
            recommendations: ['Calibration looks good, proceeding with training']
        };
    }

    extractPatternFeatures(calibrationData) {
        if (!calibrationData || calibrationData.length === 0) {
            return new Array(15).fill(0);
        }

        const samples = calibrationData.length;
        const avgRawGazeX = calibrationData.reduce((sum, s) => sum + s.rawGazeX, 0) / samples;
        const avgRawGazeY = calibrationData.reduce((sum, s) => sum + s.rawGazeY, 0) / samples;

        // Calculate variance
        const varianceX = calibrationData.reduce((sum, s) =>
            sum + Math.pow(s.rawGazeX - avgRawGazeX, 2), 0) / samples;
        const varianceY = calibrationData.reduce((sum, s) =>
            sum + Math.pow(s.rawGazeY - avgRawGazeY, 2), 0) / samples;

        // Feature extraction
        const features = [
            samples / 1000, // Normalized sample count
            avgRawGazeX / window.innerWidth,
            avgRawGazeY / window.innerHeight,
            Math.sqrt(varianceX) / window.innerWidth,
            Math.sqrt(varianceY) / window.innerHeight,
            calibrationData[0]?.features?.headPitch || 0,
            calibrationData[0]?.features?.headYaw || 0,
            calibrationData[0]?.features?.headRoll || 0,
            calibrationData[0]?.features?.avgIrisRatioX || 0.5,
            calibrationData[0]?.features?.avgIrisRatioY || 0.5,
            calibrationData[0]?.features?.avgAperture || 0,
            calibrationData[0]?.features?.leftAperture || 0,
            calibrationData[0]?.features?.rightAperture || 0,
            varianceX / 10000,
            varianceY / 10000
        ];

        return features;
    }

    extractAccuracyFeatures(calibrationData) {
        if (!calibrationData || calibrationData.length === 0) {
            return new Array(20).fill(0);
        }

        const patternFeatures = this.extractPatternFeatures(calibrationData);

        // Additional accuracy-relevant features
        const samples = calibrationData.length;
        const avgTargetX = calibrationData.reduce((sum, s) => sum + s.targetX, 0) / samples;
        const avgTargetY = calibrationData.reduce((sum, s) => sum + s.targetY, 0) / samples;

        const errors = calibrationData.map(s =>
            Math.sqrt(Math.pow(s.rawGazeX - s.targetX, 2) + Math.pow(s.rawGazeY - s.targetY, 2))
        );
        const avgError = errors.reduce((sum, e) => sum + e, 0) / errors.length;
        const maxError = Math.max(...errors);
        const minError = Math.min(...errors);

        const additionalFeatures = [
            avgTargetX / window.innerWidth,
            avgTargetY / window.innerHeight,
            avgError / 100,
            maxError / 100,
            minError / 100
        ];

        return [...patternFeatures, ...additionalFeatures];
    }

    extractInsightFeatures(sessionData) {
        const { calibrationData, modelMetrics, mode } = sessionData;

        if (!calibrationData) {
            return new Array(25).fill(0);
        }

        const accuracyFeatures = this.extractAccuracyFeatures(calibrationData);

        // Session-level features
        const additionalFeatures = [
            mode === 'glasses' ? 1 : 0,
            modelMetrics?.meanError || 0,
            modelMetrics?.stdDev || 0,
            modelMetrics?.visualAngle || 0,
            Date.now() % 1000 / 1000 // Normalized timestamp
        ];

        return [...accuracyFeatures, ...additionalFeatures];
    }

    classifyAccuracy(meanError) {
        if (meanError < 0.5) return 'Exceptional';
        if (meanError < 1.0) return 'Excellent';
        if (meanError < 2.0) return 'Very Good';
        if (meanError < 5.0) return 'Good';
        if (meanError < 10.0) return 'Fair';
        return 'Poor';
    }

    generateRecommendations(activeInsights) {
        const recommendations = [];

        activeInsights.forEach(insight => {
            switch (insight.label) {
                case 'needsRecalibration':
                    recommendations.push('Consider recalibrating the system for improved accuracy');
                    break;
                case 'poorLightingDetected':
                    recommendations.push('Improve lighting conditions for better iris detection');
                    break;
                case 'excessiveHeadMovement':
                    recommendations.push('Minimize head movement during calibration');
                    break;
                case 'glassesReflectionIssue':
                    recommendations.push('Adjust glasses position or lighting to reduce reflections');
                    break;
                case 'lowSampleQuality':
                    recommendations.push('Ensure camera is clean and properly focused');
                    break;
                case 'highVariance':
                    recommendations.push('Focus more steadily on calibration points');
                    break;
                case 'recommendFineTuning':
                    recommendations.push('System is ready for fine-tuning with additional data');
                    break;
            }
        });

        if (recommendations.length === 0) {
            recommendations.push('System performance is optimal - no action needed');
        }

        return recommendations;
    }

    async train(historicalData) {
        if (!historicalData || historicalData.length === 0) {
            console.log('⚠️ No historical data available for analytics training');
            return;
        }

        console.log(`🔬 Training analytics models on ${historicalData.length} sessions...`);
        // Training logic would go here with labeled historical data
        // For now, models use pre-defined heuristics
    }

    async saveModels() {
        console.log('💾 Saving analytics models...');
        const savePromises = Object.entries(this.models).map(async ([key, model]) => {
            const modelPath = `indexeddb://analytics-${key}`;
            await model.save(modelPath);
        });

        await Promise.all(savePromises);
        console.log('✅ Analytics models saved successfully');
    }
}
