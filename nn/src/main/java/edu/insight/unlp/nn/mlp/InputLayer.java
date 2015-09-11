package edu.insight.unlp.nn.mlp;

import java.util.Map;

import edu.insight.unlp.nn.NNLayer;

public class InputLayer implements NNLayer {

	private int numUnits;
	private int activationCounter = -1;
	private double[] activations;

	public InputLayer(int numUnits){
		this.numUnits = numUnits;
	}

	@Override
	public int numNeuronUnits() {
		return numUnits;
	}

	@Override
	public Map<Integer, double[]> lastActivations() {
		return null;
	}

	@Override
	public double[] errorGradient(double[] target) {
		return null;
	}

	@Override
	public void update(double alpha) {
	}

	@Override
	public void initializeLayer(int previousLayerUnits) {
	}

	@Override
	public double[] computeActivations(double[] input) {
		activationCounter++;
		activations = input;
		return input;
	}

	@Override
	public double[] activations() {
		return activations;
	}

	public void resetActivationCounter(){
		activationCounter = -1;
	}

	@Override
	public double[] output(double[] input) {
		activationCounter++;
		activations = input;
		return input;
	}

	@Override
	public int getActivationCounterVal() {
		return activationCounter;
	}

	@Override
	public void update(double learningRate, double momentum) {
	}

}
