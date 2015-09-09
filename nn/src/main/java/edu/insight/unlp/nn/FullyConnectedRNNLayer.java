package edu.insight.unlp.nn;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

public class FullyConnectedRNNLayer implements NNLayer {
	private int numUnits;
	private ActivationFunction af;
	private double[] weights; //keeps the weights of the connections from the previous layer
	private double[] deltas;
	private double[] prevDeltas;
	//private double[] activations; //needed by the next layer, or this layer for feedback from the last example
	//private double[] signals; //can be removed if derivatives calculated in the computeActivations method, but then derivatives need to be stored
	private Map<Integer, double[]> lastActivationDerivatives;
	private Map<Integer, double[]> lastActivations; //needed by this layer for feedback from the last example, RNN	
	private double[] nextStageError;
	private NN nn;
	private int activationCounter = -1;

	public FullyConnectedRNNLayer(int numUnits, ActivationFunction af, NN nn) {
		this.numUnits = numUnits;
		this.af = af;
		this.nn = nn;
	}

	@Override
	public int numNeuronUnits() {
		return numUnits;
	}

	public int getActivationCounterVal(){
		return activationCounter;
	}

	@Override
	public void resetActivationCounter(){
		activationCounter = -1;
		deltas = new double[weights.length];
		nextStageError = new double[numUnits + 1];
	}

	@Override
	public Map<Integer, double[]> lastActivations() {
		return lastActivations;
	}

	@Override
	public double[] activations() {
		return lastActivations.get(activationCounter);
	}

	@Override
	public double[] errorGradient(double[] eg) {
		for(int i=0; i<eg.length-1; i++){
			eg[i] = eg[i] + nextStageError[i];
		}
		int currentIndex = nn.getLayers().indexOf(this);
		if(currentIndex!=0){
			NNLayer prevLayer = nn.getLayers().get(currentIndex-1);
			Map<Integer, double[]> prevLayerActivationsMap = prevLayer.lastActivations();
			double[] prevLayerActivations = prevLayerActivationsMap.get(activationCounter);
			double[] activations = lastActivations.get(activationCounter);
			double[] derivatives = lastActivationDerivatives.get(activationCounter);
			double[] egPrevLayer = new double[prevLayerActivations.length + 1];
			double[] egPrevStage = new double[lastActivations.get(0).length + 1];
			for(int i=0; i<eg.length-1; i++){
				int currentWeightIndex = i * (1 + prevLayerActivations.length + lastActivations.get(0).length);			
				double lambda = eg[i] * derivatives[i];
				deltas[currentWeightIndex] = deltas[currentWeightIndex] +  1 * lambda; //the bias one, multiplied the weight by 1, so added directly to outputs
				int j = 0;
				for(j=0; j<prevLayerActivations.length; j++){					
					double delta = lambda * prevLayerActivations[j];
					deltas[currentWeightIndex + j + 1] = deltas[currentWeightIndex + j + 1] +  delta;
					egPrevLayer[j] = egPrevLayer[j] + delta * weights[currentWeightIndex + j + 1];
				}
				for(int m=j; m<activations.length+j; m++){					
					double delta = lambda * activations[m-j];
					deltas[currentWeightIndex + m + 1] = deltas[currentWeightIndex + m + 1] + delta;
					egPrevStage[m-j] = egPrevStage[m-j] + delta * weights[currentWeightIndex + m + 1];
				}
			}
			egPrevLayer[prevLayerActivations.length] = eg[eg.length-1];
			egPrevStage[activations.length] = eg[eg.length-1];
			nextStageError = egPrevStage;
			activationCounter--;
			return egPrevLayer;
		}
		return null;
	}

	@Override
	public void update(double learningRate) {
		IntStream.range(0, weights.length).forEach(i -> weights[i] = weights[i] - learningRate * deltas[i]);
	}

	@Override
	public void initializeLayer(int previousLayerUnits) {
		weights = new double[(previousLayerUnits+1+numUnits) * numUnits]; //+numUnits for feedback
		IntStream.range(0, weights.length).forEach(i -> weights[i] = (Math.random() * 2 - 1));
		deltas = new double[weights.length];
		prevDeltas = new double[weights.length];
		lastActivations = new HashMap<Integer, double[]>();
		lastActivationDerivatives = new HashMap<Integer, double[]>();
		lastActivations.put(-1, new double[numUnits]);
		nextStageError = new double[numUnits + 1];
	}

	@Override
	public double[] computeActivations(double[] input) {
		double[] signals = computeSignals(input);
		double[] derivatives = new double[numUnits];
		double[] activations = new double[numUnits];
		IntStream.range(0, signals.length).forEach(i -> activations[i] = af.activation(signals[i]));
		IntStream.range(0, signals.length).forEach(i -> derivatives[i] = af.activationDerivative(signals[i]));
		activationCounter++;
		lastActivations.put(activationCounter, activations);
		lastActivationDerivatives.put(activationCounter, derivatives);
		return activations;
	}

	@Override
	public double[] output(double[] input) {
		double[] signals = computeSignals(input);
		double[] activations = new double[numUnits];
		IntStream.range(0, signals.length).forEach(i -> activations[i] = af.activation(signals[i]));
		activationCounter++;
		lastActivations.put(activationCounter, activations);
		return activations;
	}

	private double[] computeSignals(double[] input){
		double outputs[] = new double[numUnits];
		for (int i = 0; i < outputs.length; i++) {
			outputs[i] = 1 * weights[i * (input.length + 1 + numUnits)]; //the bias one, multiplied the weight by 1, so added directly to outputs
			int j = 0;
			for (j = 0; j < input.length; j++) {
				outputs[i] += input[j] * weights[i * (input.length + 1 + numUnits) + j + 1];
			}
			for (int m = j; m < numUnits+j; m++) {
				outputs[i] += lastActivations.get(activationCounter)[m-j] * weights[i * (input.length + 1 + numUnits) + m + 1];
			}
		}
		return outputs;
	}

	@Override
	public void update(double learningRate, double momentum) {
		IntStream.range(0, weights.length).forEach(i -> weights[i] = weights[i] - learningRate * deltas[i] - momentum * prevDeltas[i]);
		prevDeltas  = deltas;
	}
}
