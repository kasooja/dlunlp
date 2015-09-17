package edu.insight.unlp.nn.common;

import java.util.HashMap;
import java.util.Map;

import edu.insight.unlp.nn.NNLayer;

public class InputLayer implements NNLayer {

	private int numUnits;
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
		lastActivations = new HashMap<Integer, double[]>();
	}

	@Override
	public double[] computeActivations(double[] input) {
		activationCounter++;
		lastActivations.put(activationCounter, input);
		return input;
	}

	@Override
	public double[] activations() {
		return lastActivations.get(activationCounter);
	}

	public void resetActivationCounter(){
		activationCounter = -1;
		lastActivations = new HashMap<Integer, double[]>();
	}

	@Override
	public double[] output(double[] input) {
		activationCounter++;
		lastActivations.put(activationCounter, input);
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
