package edu.insight.unlp.nn.rnn;

import java.util.Map;
import java.util.stream.IntStream;

import edu.insight.unlp.nn.ActivationFunction;
import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.NNLayer;

public class FullyConnectedRNNLayer extends NNLayer {

	private double[] nextStageError;

	public FullyConnectedRNNLayer(int numUnits, ActivationFunction af, NN nn) {
		this.numUnits = numUnits;
		this.af = af;
		this.nn = nn;
	}

	public void resetActivationCounter(boolean training){
		super.resetActivationCounter(training);
		if(training)
			nextStageError = new double[numUnits + 1];
	}

	public void initializeLayer(int previousLayerUnits){
		super.initializeLayer(previousLayerUnits, true);	
		nextStageError = new double[numUnits + 1];
	}

	public double[] errorGradient(double[] eg) {
		for(int i=0; i<eg.length-1; i++){
			eg[i] = eg[i] + nextStageError[i];
		}
		int currentIndex = nn.getLayers().indexOf(this);
		if(currentIndex!=0){
			NNLayer prevLayer = nn.getLayers().get(currentIndex-1);
			Map<Integer, double[]> prevLayerActivationsMap = prevLayer.lastActivations();
			double[] prevLayerActivations = prevLayerActivationsMap.get(activationCounter);
			double[] activations = lastActivations.get(activationCounter);
			double[] derivatives = lastActivationDerivatives.get(activationCounter);
			double[] egPrevLayer = new double[prevLayerActivations.length + 1];
			double[] egPrevStage = new double[lastActivations.get(0).length + 1];
			for(int i=0; i<eg.length-1; i++){
				int currentWeightIndex = i * (1 + prevLayerActivations.length + lastActivations.get(0).length);			
				double lambda = eg[i] * derivatives[i];
				deltas[currentWeightIndex] = deltas[currentWeightIndex] +  1 * lambda; //the bias one, multiplied the weight by 1, so added directly to outputs
				int j = 0;
				for(j=0; j<prevLayerActivations.length; j++){					
					double delta = lambda * prevLayerActivations[j];
					deltas[currentWeightIndex + j + 1] = deltas[currentWeightIndex + j + 1] +  delta;
					egPrevLayer[j] = egPrevLayer[j] + delta * weights[currentWeightIndex + j + 1];
				}
				for(int m=j; m<activations.length+j; m++){					
					double delta = lambda * activations[m-j];
					deltas[currentWeightIndex + m + 1] = deltas[currentWeightIndex + m + 1] + delta;
					egPrevStage[m-j] = egPrevStage[m-j] + delta * weights[currentWeightIndex + m + 1];
				}
			}
			lastActivations.put(activationCounter, null);
			lastActivationDerivatives.put(activationCounter, null);
			egPrevLayer[prevLayerActivations.length] = eg[eg.length-1];
			egPrevStage[activations.length] = eg[eg.length-1];
			nextStageError = egPrevStage;
			activationCounter--;
			return egPrevLayer;
		}
		return null;
	}

	public double[] computeSignals(double[] input){
		double outputs[] = new double[numUnits];
		for (int i = 0; i < outputs.length; i++) {
			outputs[i] = 1 * weights[i * (input.length + 1 + numUnits)]; //the bias one, multiplied the weight by 1, so added directly to outputs
			int j = 0;
			for (j = 0; j < input.length; j++) {
				outputs[i] += input[j] * weights[i * (input.length + 1 + numUnits) + j + 1];
			}
			for (int m = j; m < numUnits+j; m++) {
				outputs[i] += lastActivations.get(activationCounter)[m-j] * weights[i * (input.length + 1 + numUnits) + m + 1];
			}
		}
		return outputs;
	}

}
