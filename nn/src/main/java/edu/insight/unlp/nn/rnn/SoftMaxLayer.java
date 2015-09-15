package edu.insight.unlp.nn.rnn;

import java.util.HashMap;
import java.util.Map;

import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.NNLayer;

public class SoftMaxLayer implements NNLayer {
	private int numUnits;
	//private NN nn;
	private int activationCounter = -1;
	private Map<Integer, double[]> lastActivations = null;

	public SoftMaxLayer(int numUnits, NN nn) {
		this.numUnits = numUnits;
	//	this.nn = nn;
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
	}

	@Override
	public Map<Integer, double[]> lastActivations() {
		return null;
	}

	@Override
	public double[] activations() {
		return null;//lastActivations.get(activationCounter);
	}

	public double[] errorGradient(double[] eg) {
		double[] activations = lastActivations.get(activationCounter);
		double weightedErrorSum = 0.0;
		for (int i=0; i<activations.length; i++) {
			weightedErrorSum += eg[i] * activations[i];
		}
		double[] propagatedError = new double[activations.length+1];
		for (int j=0; j<activations.length; j++) {
			propagatedError[j] = (eg[j] - weightedErrorSum)*activations[j];
		}
		propagatedError[numUnits] = eg[numUnits];
		activationCounter--;
		return propagatedError;
	}

	@Override
	public void update(double learningRate) {
	}

	@Override
	public void initializeLayer(int previousLayerUnits) {
		lastActivations = new HashMap<Integer, double[]>();
		lastActivations.put(-1, new double[numUnits]);
	}

	@Override
	public double[] computeActivations(double[] input) {
		double[] activations = new double[input.length];
		double expSum = 0.0;
		for (int i=0; i<input.length; i++) {
			expSum += Math.exp(input[i]);
		}
		for (int i=0; i<activations.length; i++) {
			activations[i] = Math.exp(input[i])/expSum;
		}
		activationCounter++;
		lastActivations.put(activationCounter, activations);
		return activations;
	}

	@Override
	public double[] output(double[] input) {
		double[] activations = new double[input.length];
		double expSum = 0.0;
		for (int i=0; i<input.length; i++) {
			expSum += Math.exp(input[i]);
		}
		for (int i=0; i<activations.length; i++) {
			activations[i] = Math.exp(input[i])/expSum;
		}
		//activationCounter++;
		//lastActivations.put(activationCounter, activations);
		return activations;
	}

	@Override
	public void update(double learningRate, double momentum) {
	}
}
