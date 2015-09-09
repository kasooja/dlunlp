package edu.insight.unlp.nn;

import java.util.List;

/*
 * RNNs
 */
public class RNN implements NN {

	public List<NNLayer> layers;
	public ErrorFunction ef;
	public double[] networkOutput;

	public RNN(ErrorFunction ef){
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

	
	public void update(double learningRate, double momentum) {
		for(NNLayer layer : layers){
			int index = layers.indexOf(layer);
			if(index!=layers.size()-1){
				layers.get(index+1).update(learningRate, momentum);
			}
		}
	}

	@Override
	public double error(double[] inputs, double[] target) {
		return 0;
	}

	public double batchgdTrain(double[][] inputs, double[][] targets, double learningRate, int batchSize, boolean shuffle){
		return 0.0;
	}

	public double sgdTrainSeq(List<SequenceM21> training, double learningRate, int batchSize, boolean shuffle, double momentum){
		double overallError = 0.0;
		for(SequenceM21 seq : training){
			double[][] inputSeq = seq.inputSeq;
			double[] target = seq.target;
			double[] networkOutput = ff(inputSeq);
			double[] eg = ef.error(networkOutput, target);
			eg = bp(eg);
			double error = eg[networkOutput.length];
			overallError = overallError + error;
			update(learningRate, momentum);
			resetActivationCounter();
		}
		return overallError / training.size();
	}

	private double[] ff(double[][] inputSeq){
		double[] activations = null;
		for(double[] input : inputSeq){
			activations = input;		
			for(NNLayer layer : layers){
				activations = layer.computeActivations(activations);
			}
		}
		networkOutput = activations;
		return networkOutput;
	}

	private double[] bp(double[] errorGradient){
		int o = errorGradient.length;
		for(int j=layers.get(layers.size()-1).getActivationCounterVal(); j>=0; j--){
			for(int i = layers.size() - 1; i>0; i--){
				errorGradient = layers.get(i).errorGradient(errorGradient);
			}
			double totalError = errorGradient[errorGradient.length-1];
			errorGradient = new double[o]; errorGradient[o-1] = totalError;
		}
		return errorGradient;
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
		layers.get(0).initializeLayer(-1);
	}

	@Override
	public List<NNLayer> getLayers() {
		return layers;
	}

	@Override
	public double sgdTrain(double[][] inputs, double[][] targets,
			double learningRate) {
		return 0;
	}

	@Override
	public double[] outputSequence(double[][] inputSeq) {
		double[] result = null;
		for(double[] input : inputSeq){
			result = input;		
			for(NNLayer layer : layers){
				result = layer.output(result);
			}
		}
		networkOutput = result;
		return networkOutput;
	}

}
