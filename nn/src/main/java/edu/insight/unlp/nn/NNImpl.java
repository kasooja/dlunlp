package edu.insight.unlp.nn;

import java.util.Collections;
import java.util.List;

import edu.insight.unlp.nn.ErrorFunction;
import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.NNLayer;
import edu.insight.unlp.nn.common.Sequence;

public class NNImpl implements NN {

	protected List<NNLayer> layers;
	protected ErrorFunction ef;

	private double totalLoss = 0.0;
	private double totalSteps = 0.0;

	public NNImpl(ErrorFunction ef){
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

	public double sgdTrain(List<Sequence> training, double learningRate, boolean shuffle){
		if(shuffle){
			Collections.shuffle(training);
		}
		totalLoss = 0.0;
		totalSteps = 0.0;
		resetActivationCounter(true);
		for(Sequence seq : training){
			double[][] eg = ffError(seq);//new double[networkOutput.length][];
			bp(eg);
			update(learningRate);
			resetActivationCounter(true);
		}
		return totalLoss/totalSteps;
	}

	private double[][] ffError(Sequence seq){
		double[][] eg = new double[seq.inputSeq.length][];
		int i = 0;
		for(double[] input : seq.inputSeq){
			double[] activations = null;
			activations = input;		
			for(NNLayer layer : layers){
				activations = layer.computeActivations(activations, true);
			}
			eg[i] = ef.error(seq.target[i], activations);
			totalLoss = totalLoss + eg[i][eg[i].length-1];
			totalSteps++;
			i++;
		}
		return eg;
	}

	private void bp(double[][] errorGradient){
		int o = errorGradient[0].length-1;
		double[] stageErrorGradient = null;
		for(int j=layers.get(layers.size()-1).getActivationCounterVal(); j>=0; j--){
			stageErrorGradient = errorGradient[j];
			for(int i = layers.size() - 1; i>=0; i--){
				stageErrorGradient = layers.get(i).errorGradient(stageErrorGradient);
			}
			double totalError = stageErrorGradient[stageErrorGradient.length-1];
			if(j-1>-1){
				errorGradient[j-1][o] = totalError;
			}
		}
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
	public double[][] output(double[][] inputSeq) {
		double[][] target = new double[inputSeq.length][];
		double[] result = null;
		int i = 0;
		for(double[] input : inputSeq){
			result = input;		
			for(NNLayer layer : layers){
				result = layer.computeActivations(result, false);
			}
			target[i++] = result;
		}
		return target;
	}

}
