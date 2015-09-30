package edu.insight.unlp.nn.mlp;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import edu.insight.unlp.nn.ErrorFunction;
import edu.insight.unlp.nn.MLP;
import edu.insight.unlp.nn.NNLayer;

/*
 * Feedforward MLP
 */
public class MLPImpl implements MLP {

	public List<NNLayer> layers;
	public ErrorFunction ef;

	public MLPImpl(ErrorFunction ef){
		this.ef = ef;
	}

	@Override
	public void update(double learningRate) {
		for(NNLayer layer : layers){
			int index = layers.indexOf(layer);
			if(index!=layers.size()-1){
				layers.get(index+1).update(learningRate);
			}
		}
	}

	public double sgdTrain(double[][] inputs, double[][] targets, double learningRate, boolean shuffle){
		int batchSize = 1; // only sgd, currently batchgd not working as activation counter is taking care of the steps in the sequence, need another counter for continuos averaging
		resetActivationCounter(true);
		double overallError = 0.0;
		int j = 0;
		int inputSize = inputs.length;
		List<Integer> inputIndexList = new ArrayList<Integer>();
		for(int i = 0; i<inputSize; i++){
			inputIndexList.add(i);
		}
		if(shuffle){
			Collections.shuffle(inputIndexList);
		}
		for(int i : inputIndexList){
			if(j>=batchSize){
				update(learningRate);
				resetActivationCounter(true);
				j=0;
			}					
			double[] networkOutput = ff(inputs[i], true);
			double[] eg = ef.error(targets[i], networkOutput);
			eg = bp(eg);
			double error = eg[inputs[i].length];
			overallError = overallError + error;
			j++;
		}
		resetActivationCounter(false);		
		return overallError / inputs.length;
	}

	private double[] ff(double[] input, boolean training){
		double[] networkOutput = input;		
		for(NNLayer layer : layers){
			networkOutput = layer.computeActivations(networkOutput, training);
		}
		return networkOutput;
	}

	private double[] bp(double[] errorGradient){
		for(int i = layers.size() - 1; i>=0; i--){
			errorGradient = layers.get(i).errorGradient(errorGradient);			
		}
		return errorGradient;
	}

	public void resetActivationCounter(boolean training){
		for(NNLayer layer : layers){
			layer.resetActivationCounter(training);
		}
	}

	@Override
	public int numOutputUnits() {
		return layers.get(layers.size()-1).numNeuronUnits();
	}

	@Override
	public void setLayers(List<edu.insight.unlp.nn.NNLayer> layers) {
		this.layers = layers;
	}

	@Override
	public void initializeNN() {
		int prevLayerUnits = -1;
		for(int i=0; i<layers.size(); i++){
			layers.get(i).initializeLayer(prevLayerUnits);
			prevLayerUnits = layers.get(i).numNeuronUnits();
		}
	}

	@Override
	public List<NNLayer> getLayers() {
		return layers;
	}

	@Override
	public double[] output(double[] input) {
		return ff(input, false);
	}

}
