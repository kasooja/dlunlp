package edu.insight.unlp.nn;

import java.util.HashMap;
import java.util.Map;

public class InputLayer implements NNLayer {

	private int numUnits;
	private double[] activations;
	private int activationCounter = -1;
	private Map<Integer, double[]> lastActivations; //needed by this layer for feedback from the last example, RNN

	public InputLayer(int numUnits){
		this.numUnits = numUnits;
	}

	@Override
	public int numNeuronUnits() {
		return numUnits;
	}

	@Override
	public Map<Integer, double[]> lastActivations() {
		return lastActivations;
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
		activations = new double[numUnits];
		lastActivations = new HashMap<Integer, double[]>();
	}

	@Override
	public double[] computeActivations(double[] input) {
		activations = input;
		activationCounter++;
		lastActivations.put(activationCounter, activations);
		return activations;
	}

	@Override
	public double[] activations() {
		return activations;
	}

	public void resetActivationCounter(){
		activationCounter = 0;
	}

	@Override
	public double[] output(double[] input) {
		activations = input;
		return activations;
	}

	@Override
	public int getActivationCounterVal() {
		return activationCounter;
	}

}
