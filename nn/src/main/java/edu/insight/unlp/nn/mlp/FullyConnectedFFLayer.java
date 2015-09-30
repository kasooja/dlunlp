package edu.insight.unlp.nn.mlp;
import java.util.Map;

import edu.insight.unlp.nn.ActivationFunction;
import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.NNLayer;
import edu.insight.unlp.nn.common.WeightMatrix;

public class FullyConnectedFFLayer extends NNLayer {

	public FullyConnectedFFLayer(int numUnits, ActivationFunction af, NN nn) {
		this.numUnits = numUnits;
		this.af = af;
		this.nn = nn;
		weightMatrix = new WeightMatrix();
	}

	@Override
	public double[] errorGradient(double[] eg) {
		int currentIndex = nn.getLayers().indexOf(this);
		if(currentIndex!=0){
			NNLayer prevLayer = nn.getLayers().get(currentIndex-1);
			double[] prevLayerActivations = prevLayer.activations();
			double[] egPrevLayer = new double[prevLayerActivations.length + 1];
			double[] deriv = lastActivationDerivatives.get(activationCounter);
			for(int i=0; i<eg.length-1; i++){
				int currentWeightIndex = i * (prevLayerActivations.length + 1);
				double lambda = eg[i] * deriv[i];//af.activationDerivative(signals[i]);
				//deltas[currentWeightIndex] = (deltas[currentWeightIndex] * activationCounter + 1 * lambda) / (activationCounter + 1); //the bias one, multiplied the weight by 1, so added directly to outputs
				weightMatrix.deltas[currentWeightIndex] = weightMatrix.deltas[currentWeightIndex] + 1 * lambda; //the bias one, multiplied the weight by 1, so added directly to outputs
				for(int j=0; j<prevLayerActivations.length; j++){					
					double delta = lambda * prevLayerActivations[j];
					//deltas[currentWeightIndex + j + 1] = (deltas[currentWeightIndex + j + 1] * activationCounter + delta) / (activationCounter+1);
					weightMatrix.deltas[currentWeightIndex + j + 1] = weightMatrix.deltas[currentWeightIndex + j + 1] + delta;
					egPrevLayer[j] = egPrevLayer[j] + lambda * weightMatrix.weights[currentWeightIndex + j + 1];
				}
			}
			lastActivations.put(activationCounter, null);
			lastActivationDerivatives.put(activationCounter, null);
			egPrevLayer[prevLayerActivations.length] = eg[eg.length-1];
			activationCounter--;
			return egPrevLayer;
		} else {
			activationCounter--;
			return eg;
		}
	}

	public double[] computeSignals(double[] input, WeightMatrix weightMatrix, Map<Integer, double[]> activations){ //MLP does not require its activations for feedback, so not used in the method for computing signals
		if(prevLayerUnits == -1){
			return input;
		}
		double outputs[] = new double[numUnits];
		for (int i = 0; i < outputs.length; i++) {
			outputs[i] = 1 * weightMatrix.weights[i * (input.length + 1)]; //the bias one, multiplied the weight by 1, so added directly to outputs
			for (int j = 0; j < input.length; j++) {
				outputs[i] += input[j] * weightMatrix.weights[i * (input.length + 1) + j + 1];
			}
		}
		return outputs;
	}

	@Override
	public void initializeLayer(int previousLayerUnits) {
		super.initializeLayer(previousLayerUnits, false);
	}

}
