package edu.insight.unlp.nn.lstm;

import java.util.Map;

import edu.insight.unlp.nn.NNLayer;

public class FullyConnectedLSTMLayer implements NNLayer {

	@Override
	public int numNeuronUnits() {
		return 0;
	}

	@Override
	public Map<Integer, double[]> lastActivations() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public double[] activations() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public double[] errorGradient(double[] input) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void update(double learningRate) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void update(double learningRate, double momentum) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public double[] computeActivations(double[] input) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public double[] output(double[] input) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void initializeLayer(int previousLayerUnits) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void resetActivationCounter() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public int getActivationCounterVal() {
		// TODO Auto-generated method stub
		return 0;
	}
	
	

}
