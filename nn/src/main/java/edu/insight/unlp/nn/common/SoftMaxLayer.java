package edu.insight.unlp.nn.common;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

import org.apache.commons.math.random.RandomDataImpl;

import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.NNLayer;

public class SoftMaxLayer implements NNLayer {
	private int numUnits;
	private NN nn;
	private int activationCounter = -1;
	private double[] weights; //keeps the weights of the connections from the previous layer
	private Map<Integer, double[]> lastActivations = null;
	private double[] prevDeltas;
	private double[] deltas;


	public SoftMaxLayer(int numUnits, NN nn) {
		this.numUnits = numUnits;
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
		int currentIndex = nn.getLayers().indexOf(this);
		NNLayer prevLayer = nn.getLayers().get(currentIndex-1);
		double[] prevLayerActivations = prevLayer.lastActivations().get(activationCounter);
		double[] egPrevLayer = new double[prevLayer.numNeuronUnits()+1];
		for(int i=0; i<eg.length-1; i++){
			int currentWeightIndex = i * (prevLayer.numNeuronUnits() + 1);			
			double lambda = eg[i] * 1;
			deltas[currentWeightIndex] =  1 * lambda; //the bias one, multiplied the weight by 1, so added directly to outputs
			for(int j=0; j<prevLayerActivations.length; j++){					
				double delta = lambda * prevLayerActivations[j];
				deltas[currentWeightIndex + j + 1] = delta;
				egPrevLayer[j] = egPrevLayer[j] + delta * weights[currentWeightIndex + j + 1];
			}
		}
		egPrevLayer[prevLayerActivations.length] = eg[eg.length-1]; 
		activationCounter--;
		return egPrevLayer;
	}

	@Override
	public void update(double learningRate) {
	}

	@Override
	public void initializeLayer(int previousLayerUnits) {
		lastActivations = new HashMap<Integer, double[]>();
		lastActivations.put(-1, new double[numUnits]);
		weights = new double[(previousLayerUnits+1) * numUnits];
		double eInit = Math.sqrt(6) / Math.sqrt(numUnits + previousLayerUnits);
		setWeightsUniformly(seedRandomGenerator(), eInit);
		deltas = new double[weights.length];
		prevDeltas = new double[weights.length];
	}

	@Override
	public double[] computeActivations(double[] input) {
		double[] signals = computeSignals(input);
		double expSum = 0.0;
		for (int i=0; i<signals.length; i++) {
			signals[i] = Math.exp(signals[i]); 
			expSum += signals[i];
		}
		for (int i=0; i<signals.length; i++) {
			signals[i] = signals[i]/expSum;
		}
		activationCounter++;
		lastActivations.put(activationCounter, signals);
		return signals;
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
	public double[] output(double[] input) {
		double[] signals = computeSignals(input);
		double[] activations = new double[signals.length];
		double expSum = 0.0;
		for (int i=0; i<signals.length; i++) {
			expSum += Math.exp(signals[i]);
		}
		for (int i=0; i<activations.length; i++) {
			activations[i] = Math.exp(signals[i])/expSum;
		}
		//activationCounter++;
		//lastActivations.put(activationCounter, activations);
		return activations;
	}

	@Override
	public void update(double learningRate, double momentum) {
		IntStream.range(0, weights.length).forEach(i -> weights[i] = weights[i] - learningRate * deltas[i] - momentum * prevDeltas[i]);
		prevDeltas = deltas;
	}

	/**
	 * Sets the weights in the whole matrix uniformly between -eInit and eInit
	 * (eInit is the standard deviation) with zero mean.
	 */
	private void setWeightsUniformly(RandomDataImpl rnd, double eInit) {
		for (int i = 0; i < weights.length; i++) {		
			weights[i] = rnd.nextUniform(-eInit, eInit);
		}
	}

	private RandomDataImpl seedRandomGenerator() {
		RandomDataImpl rnd = new RandomDataImpl();
		rnd.reSeed(System.currentTimeMillis());
		rnd.reSeedSecure(System.currentTimeMillis());
		return rnd;
	}

}
