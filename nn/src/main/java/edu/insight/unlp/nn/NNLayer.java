package edu.insight.unlp.nn;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

import edu.insight.unlp.nn.common.WeightInitializer;

public abstract class NNLayer {

	protected NN nn;
	protected int activationCounter = -1;
	protected ActivationFunction af;
	protected Map<Integer, double[]> lastActivationDerivatives;
	protected Map<Integer, double[]> lastActivations; //needed by this layer for feedback from the last example, RNN	
	protected int numUnits;
	protected int prevLayerUnits;
	protected double[] weights; //keeps the weights of the connections from the previous layer
	protected double[] deltas;
	protected double[] prevDeltas; // stepCache, explore more
	public static double decayRate = 0.999;
	public static double smoothEpsilon = 1e-8;
	public static double gradientClipValue = 5;
	public static double regularization = 0.000001; // L2 regularization strength

	public int numNeuronUnits(){
		return numUnits;
	}

	public Map<Integer, double[]> lastActivations() {
		return lastActivations;
	}

	public double[] activations() {
		return lastActivations.get(activationCounter);
	}

	public void resetActivationCounter(boolean training){
		activationCounter = -1;
		if(training && prevLayerUnits!=-1){
			deltas = new double[weights.length];
			prevDeltas = new double[weights.length];
		}
	}


	public void update(double learningRate) {
		for(int i=0; i<weights.length; i++) {
			double mdwi = deltas[i]; // rmsprop adaptive learning rate
			prevDeltas[i] = prevDeltas[i] * decayRate + (1 - decayRate) * mdwi * mdwi; 

			if (mdwi > gradientClipValue) {			// gradient clip
				mdwi = gradientClipValue;
			}
			if (mdwi < -gradientClipValue) {
				mdwi = -gradientClipValue;
			}
			// update (and regularize)
			weights[i] += - learningRate * mdwi / Math.sqrt(prevDeltas[i] + smoothEpsilon) - regularization * weights[i];
		}
		//	IntStream.range(0, weights.length).forEach(i -> weights[i] = weights[i] - learningRate * deltas[i] - momentum * prevDeltas[i]);
		//	prevDeltas  = deltas;
	}

	public double[] computeActivations(double[] input, boolean training) {
		double[] signals = computeSignals(input);
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
		int totalWeightParams = (previousLayerUnits+1) * numUnits;;
		if(feedback)
			totalWeightParams = (previousLayerUnits+1+numUnits) * numUnits;
		weights = new double[totalWeightParams];
		WeightInitializer.randomInitializeLeCun(weights);//(weights, 0.2);
		deltas = new double[weights.length];
		prevDeltas = new double[weights.length];
		lastActivationDerivatives = new HashMap<Integer, double[]>();
	}
	
	public int getActivationCounterVal(){
		return activationCounter;
	}

	public abstract double[] errorGradient(double[] input);

	public abstract double[] computeSignals(double[] input);
	
	public abstract void initializeLayer(int previousLayerUnits);


}