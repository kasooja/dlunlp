package edu.insight.unlp.nn;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

import edu.insight.unlp.nn.common.WeightInitializer;
import edu.insight.unlp.nn.common.WeightMatrix;

public abstract class NNLayer implements Serializable {

	private static final long serialVersionUID = 1L;
	protected NN nn;
	protected int overallNNOutputUnits; 
	protected int activationCounter = -1;
	protected ActivationFunction af;
	protected Map<Integer, double[]> lastActivationDerivatives;
	protected Map<Integer, double[]> lastActivations; //needed by this layer for feedback in RNNs, it keeps the lstm block output activations as feedback  	
	protected int numUnits; 
	protected int prevLayerUnits;
	protected WeightMatrix weightMatrix;
	public static double decayRate = 0.999;
	public static double smoothEpsilon = 1e-8;
	public static double gradientClipValue = 5;
	public static double regularization = 0.000001; // L2 regularization strength
	public static double initParamsStdDev = 0.08;

	public int numNeuronUnits(){
		return numUnits;
	}

	public Map<Integer, double[]> lastActivations() {
		return lastActivations;
	}

	public void resetActivationCounter(boolean training){
		activationCounter = -1;
	}

	public void update(double learningRate, WeightMatrix weightMatrix){
		for(int i=0; i<weightMatrix.weights.length; i++) {
			double mdwi = weightMatrix.deltas[i]; // rmsprop adaptive learning rate
			weightMatrix.stepCache[i] = weightMatrix.stepCache[i] * decayRate + (1 - decayRate) * mdwi * mdwi; 
			if (mdwi > gradientClipValue) {			// gradient clip
				mdwi = gradientClipValue;
			}
			if (mdwi < -gradientClipValue) {
				mdwi = -gradientClipValue;
			}
			// update (and regularize)
			weightMatrix.weights[i] += - learningRate * mdwi / Math.sqrt(weightMatrix.stepCache[i] + smoothEpsilon) - regularization * weightMatrix.weights[i];
			weightMatrix.deltas[i] = 0;
		}
		//	IntStream.range(0, weights.length).forEach(i -> weights[i] = weights[i] - learningRate * deltas[i] - momentum * prevDeltas[i]);
		//	prevDeltas  = deltas;
	}

	public void update(double learningRate){
		update(learningRate, weightMatrix);
	}

	public double[] computeActivations(double[] input, boolean training) {
		double[] signals = computeSignals(input, weightMatrix, lastActivations);
		double[] derivatives = new double[numUnits];
		double[] activations = new double[numUnits];
		activationCounter++;
		IntStream.range(0, signals.length).forEach(i -> activations[i] = af.activation(signals[i]));
		lastActivations.put(activationCounter, activations);
		if(training && prevLayerUnits != -1){
			IntStream.range(0, signals.length).forEach(i -> derivatives[i] = af.activationDerivative(signals[i]));
			lastActivationDerivatives.put(activationCounter, derivatives);
		}
		return activations;
	}

	public void initializeLayer(int previousLayerUnits, boolean feedback){
		this.prevLayerUnits = previousLayerUnits;
		lastActivations = new HashMap<Integer, double[]>();
		lastActivations.put(-1, new double[numUnits]);
		if(prevLayerUnits==-1){
			return;
		}
		initializeLayerWeights(weightMatrix, feedback, 0.0);
		lastActivationDerivatives = new HashMap<Integer, double[]>();
	}
	
	public void initializeLayer(int previousLayerUnits, boolean ownFeedback, boolean prevOutputFeedback){
		overallNNOutputUnits = nn.getLayers().get(nn.getLayers().size()-1).numUnits;
		this.prevLayerUnits = previousLayerUnits;
		lastActivations = new HashMap<Integer, double[]>();
		lastActivations.put(-1, new double[numUnits]);
		if(prevLayerUnits==-1){
			return;
		}
		initializeLayerWeights(weightMatrix, ownFeedback, 0.0, true);
		lastActivationDerivatives = new HashMap<Integer, double[]>();
	}

	protected void initializeLayerWeights(WeightMatrix weightMatrix, boolean feedback, double biasWeight){
		int totalWeightParams = (prevLayerUnits+1) * numUnits;
		weightMatrix.biasMultiplier = (prevLayerUnits+1);
		if(feedback){
			totalWeightParams = (prevLayerUnits+1+numUnits) * numUnits;
			weightMatrix.biasMultiplier = (prevLayerUnits+1+numUnits);
		}
		weightMatrix.weights = new double[totalWeightParams];
		weightMatrix.deltas = new double[totalWeightParams];
		weightMatrix.stepCache = new double[totalWeightParams];
		initializeLayerWeights(weightMatrix, biasWeight);
	}
	
	protected void initializeLayerWeights(WeightMatrix weightMatrix, boolean ownFeedback, double biasWeight, boolean outputFeedback){
		int totalWeightParams = (prevLayerUnits+1) * numUnits;
		weightMatrix.biasMultiplier = (prevLayerUnits+1);
		if(ownFeedback){
			totalWeightParams = (prevLayerUnits+1+numUnits) * numUnits;
			weightMatrix.biasMultiplier = (prevLayerUnits+1+numUnits);
		}
		if(ownFeedback && outputFeedback){
			totalWeightParams = (prevLayerUnits+1+numUnits+overallNNOutputUnits) * numUnits;
			weightMatrix.biasMultiplier = (prevLayerUnits+1+numUnits+overallNNOutputUnits);
		}
		weightMatrix.weights = new double[totalWeightParams];
		weightMatrix.deltas = new double[totalWeightParams];
		weightMatrix.stepCache = new double[totalWeightParams];
		initializeLayerWeights(weightMatrix, biasWeight);
	}


	protected void initializeLayerWeights(WeightMatrix weightMatrix, double biasWeight) {
		WeightInitializer.randomInitializeKarapathyCode(weightMatrix, initParamsStdDev, biasWeight);
		//WeightInitializer.constantInitialize(weightMatrix, 0.2, null);
		//all other biases to zero in every case, set forget bias to 1.0, as described here: http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf
		//WeightInitializer.randomInitializeLeCun(weightMatrix, null);
	}

	public int getActivationCounterVal(){
		return activationCounter;
	}

	public abstract double[] errorGradient(double[] input);

	/*
	 * testing for NA
	 */
	public abstract double[] errorGradient(double[] eg, double[] input, double[] na);

	public abstract double[] computeSignals(double[] input, WeightMatrix weights, Map<Integer, double[]> activations);
	public abstract void initializeLayer(int previousLayerUnits);

}