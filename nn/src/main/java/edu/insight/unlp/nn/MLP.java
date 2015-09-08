package edu.insight.unlp.nn;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/*
 * Feedforward MLP
 */
public class MLP implements NN {

	public List<NNLayer> layers;
	public ErrorFunction ef;
	public double[] networkOutput;

	public MLP(ErrorFunction ef){
		this.ef = ef;
	}

	public double[] output(double[] input) {
		double[] result = input;
		for(NNLayer nnlayer : layers){
			result = nnlayer.output(result);
		}
		return result;
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

	@Override
	public double error(double[] inputs, double[] target) {
		return 0;
	}

	public double batchgdTrain(double[][] inputs, double[][] targets, double learningRate, int batchSize, boolean shuffle){
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
				resetActivationCounter();
				update(learningRate);
				j=0;
			}					
			double[] networkOutput = ff(inputs[i]);
			double[] eg = ef.error(networkOutput, targets[i]);
			eg = bp(eg);
			double error = eg[inputs[i].length];
			overallError = overallError + error;
			j++;
		}
		return overallError / inputs.length;
	}

	private double[] ff(double[] input){
		double[] activations = input;		
		for(NNLayer layer : layers){
			activations = layer.computeActivations(activations);
		}
		networkOutput = activations;
		return networkOutput;
	}

	private double[] bp(double[] errorGradient){
		for(int i = layers.size() - 1; i>0; i--){
			errorGradient = layers.get(i).errorGradient(errorGradient);			
		}
		return errorGradient;
	}

	public double sgdTrain(double[][] inputs, double[][] targets, double learningRate) {		
		int inputSize = inputs.length;
		List<Integer> inputIndexList = new ArrayList<Integer>();
		for(int i = 0; i<inputSize; i++){
			inputIndexList.add(i);
		}
		Collections.shuffle(inputIndexList);
		double overallError = 0.0;
		for(int i : inputIndexList){
			double[] networkOutput = ff(inputs[i]);
			double[] eg = ef.error(networkOutput, targets[i]);
			eg = bp(eg);
			double error = eg[inputs[i].length];
			overallError = overallError + error;
			update(learningRate);	
			resetActivationCounter();
		}
		return overallError / inputs.length;
	}

	public void resetActivationCounter(){
		for(NNLayer layer : layers){
			layer.resetActivationCounter();
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
		for(NNLayer layer : layers){
			int index = layers.indexOf(layer);
			if(index!=layers.size()-1){
				layers.get(index+1).initializeLayer(layer.numNeuronUnits());
			}
		}
	}

	@Override
	public List<NNLayer> getLayers() {
		return layers;
	}

	@Override
	public double[] outputSequence(double[][] inputSeq) {
		return null;
	}

}
