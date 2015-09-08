package edu.insight.unlp.nn;

import java.util.Map;
import java.util.stream.IntStream;

public class FullyConnectedLayer implements NNLayer {

	private int numUnits;
	private ActivationFunction af;
	private double[] weights; //keeps the weights of the connections from the previous layer
	private double[] deltas;
	private double[] activations; //needed by the next layer
	private double[] signals; // can be removed if derivatives calculated in the computeActivations method, but then derivatives need to be stored 
	private NN nn;
	private int activationCounter = -1;

	public FullyConnectedLayer(int numUnits, ActivationFunction af, NN nn) {
		this.numUnits = numUnits;
		this.af = af;
		this.nn = nn;
	}

	@Override
	public int numNeuronUnits() {
		return numUnits;
	}
	
	@Override
	public void resetActivationCounter(){
		activationCounter = -1;
	}

	@Override
	public Map<Integer, double[]> lastActivations() {
		return null;
	}

	@Override
	public double[] activations() {
		return activations;
	}

	@Override
	public double[] errorGradient(double[] eg) {
		int currentIndex = nn.getLayers().indexOf(this);
		if(currentIndex!=0){
			NNLayer prevLayer = nn.getLayers().get(currentIndex-1);
			double[] prevLayerActivations = prevLayer.activations();
			double[] egPrevLayer = new double[prevLayerActivations.length + 1];
			for(int i=0; i<eg.length-1; i++){
				int currentWeightIndex = i * (prevLayerActivations.length + 1);			
				double lambda = eg[i] * af.activationDerivative(signals[i]);
				deltas[currentWeightIndex] = (deltas[currentWeightIndex] * activationCounter + 1 * lambda) / (activationCounter + 1); //the bias one, multiplied the weight by 1, so added directly to outputs
				for(int j=0; j<prevLayerActivations.length; j++){					
					double delta = lambda * prevLayerActivations[j];
					deltas[currentWeightIndex + j + 1] = (deltas[currentWeightIndex + j + 1] * activationCounter + delta) / (activationCounter+1);
					egPrevLayer[j] = egPrevLayer[j] + delta * weights[currentWeightIndex + j + 1];
				}
			}
			egPrevLayer[prevLayerActivations.length] = eg[eg.length-1]; 
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
		weights = new double[(previousLayerUnits+1) * numUnits];
		IntStream.range(0, weights.length).forEach(i -> weights[i] = (Math.random() * 2 - 1));
		deltas = new double[weights.length];
		activations = new double[numUnits];
	}

	@Override
	public double[] computeActivations(double[] input) {
		output(input);
		activationCounter++;
		return activations;
	}

	@Override
	public double[] output(double[] input) {
		signals = computeSignals(input);			
		IntStream.range(0, signals.length).forEach(i -> activations[i] = af.activation(signals[i]));
		return activations;
	}
	
	private double[] computeSignals(double[] input){
		double outputs[] = new double[numUnits];
		for (int i = 0; i < outputs.length; i++) {
			outputs[i] = 1 * weights[i * (input.length + 1)]; //the bias one, multiplied the weight by 1, so added directly to outputs
			for (int j = 0; j < input.length; j++) {
				outputs[i] += input[j] * weights[i * (input.length + 1) + j + 1];
			}
		}
		return outputs;
	}

	@Override
	public int getActivationCounterVal() {
		return activationCounter;
	}

}
