package edu.insight.unlp.nn.layers;

import java.util.Map;

import edu.insight.unlp.nn.ActivationFunction;
import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.NNLayer;
import edu.insight.unlp.nn.common.WeightMatrix;

public class NAFCRNNLayer extends NNLayer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private double[] nextStageError;
	private double[] nextOutputStageError;

	public NAFCRNNLayer(int numUnits, ActivationFunction af, NN nn) {
		this.numUnits = numUnits;
		this.af = af;
		this.nn = nn;
		weightMatrix = new WeightMatrix();
	}

	public void resetActivationCounter(boolean training){
		super.resetActivationCounter(training);
		if(training){
			nextStageError = new double[numUnits + 1];
			nextOutputStageError = new double[overallNNOutputUnits + 1];
		}
	}

	public void initializeLayer(int previousLayerUnits){
		super.initializeLayer(previousLayerUnits, true, true);
		nextStageError = new double[numUnits + 1];
		nextOutputStageError = new double[overallNNOutputUnits + 1];
	}

	public double[] errorGradient(double[] eg) {
		for(int i=0; i<eg.length-1; i++){
			eg[i] = eg[i] + nextStageError[i];
		}
		double[] egPrevLayer = eg;
		int currentIndex = nn.getLayers().indexOf(this);
		if(currentIndex!=0){
			NNLayer prevLayer = nn.getLayers().get(currentIndex-1);
			Map<Integer, double[]> prevLayerActivationsMap = prevLayer.lastActivations();
			double[] prevLayerActivations = prevLayerActivationsMap.get(activationCounter);
			double[] feedbackActivations = lastActivations.get(activationCounter-1);
			double[] outputActivations = nn.getLayers().get(nn.getLayers().size()-1).lastActivations().get(activationCounter-1);
			double[] derivatives = lastActivationDerivatives.get(activationCounter);
			egPrevLayer = new double[prevLayerActivations.length + 1];
			double[] egPrevStage = new double[lastActivations.get(0).length + 1];
			double[] egPrevStageOutput = new double[overallNNOutputUnits + 1];
			for(int i=0; i<eg.length-1; i++){
				int currentWeightIndex = i * (1 + prevLayerUnits + numUnits + overallNNOutputUnits);			
				double lambda = eg[i] * derivatives[i];
				weightMatrix.deltas[currentWeightIndex] = weightMatrix.deltas[currentWeightIndex] +  1 * lambda; //the bias one, multiplied the weight by 1, so added directly to outputs
				int j = 0;
				for(j=0; j<prevLayerUnits; j++){					
					double delta = lambda * prevLayerActivations[j];
					weightMatrix.deltas[currentWeightIndex + j + 1] = weightMatrix.deltas[currentWeightIndex + j + 1] +  delta;
					egPrevLayer[j] = egPrevLayer[j] + lambda * weightMatrix.weights[currentWeightIndex + j + 1];
				}
				int m = j;
				for(m=j; m<numUnits+j; m++){					
					double delta = lambda * feedbackActivations[m-j];
					weightMatrix.deltas[currentWeightIndex + m + 1] = weightMatrix.deltas[currentWeightIndex + m + 1] + delta;
					egPrevStage[m-j] = egPrevStage[m-j] + lambda * weightMatrix.weights[currentWeightIndex + m + 1];
				}
				for(int n=m; n<overallNNOutputUnits+m; n++){					
					double delta = lambda * outputActivations[n-m];
					weightMatrix.deltas[currentWeightIndex + n + 1] = weightMatrix.deltas[currentWeightIndex + n + 1] + delta;
					egPrevStageOutput[n-m] = egPrevStageOutput[n-m] + lambda * weightMatrix.weights[currentWeightIndex + n + 1];
				}
			}
			egPrevLayer[prevLayerUnits] = eg[eg.length-1];
			egPrevStage[numUnits] = eg[eg.length-1];
			egPrevStageOutput[overallNNOutputUnits] = eg[eg.length-1];
			nextStageError = egPrevStage;
			nextOutputStageError = egPrevStageOutput;
			
			
		} 
		resetActivationAndDerivatives(activationCounter);
		activationCounter--;
		return concat(egPrevLayer, nextOutputStageError);
	}

	
	public static double[] concat(double[]... arrays) {
		int length = 0;
		for (double[] array : arrays) {
			length += array.length;
		}
		double[] result = new double[length];
		int pos = 0;
		for (double[] array : arrays) {
			for (double element : array) {
				result[pos] = element;
				pos++;
			}
		}
		return result;
	}

	private void resetActivationAndDerivatives(int activationCounter){
		lastActivations.put(activationCounter, null);
		if(prevLayerUnits!=-1)
			lastActivationDerivatives.put(activationCounter, null);
	}

	public double[] computeSignals(double[] input, WeightMatrix weightMatrix, Int2ObjectMap<double[]> activations) {
		double signals[] = new double[numUnits];
		int multiplierSize = input.length + 1 + numUnits + overallNNOutputUnits;
		Map<Integer, double[]> lastOutputActivations = nn.getLayers().get(nn.getLayers().size()-1).lastActivations();
		for (int i = 0; i < signals.length; i++) {
			signals[i] = 1 * weightMatrix.weights[i * (multiplierSize)]; //the bias one, multiplied the weight by 1, so added directly to outputs
			int j = 0;
			for (j = 0; j < input.length; j++) {
				signals[i] += input[j] * weightMatrix.weights[i * (multiplierSize) + j + 1];
			}
			int m = j;
			for (m = j; m < numUnits+j; m++) {
				signals[i] += activations.get(activationCounter)[m-j] * weightMatrix.weights[i * (multiplierSize) + m + 1];
			}
			int n = m;
			for (n = m; n < overallNNOutputUnits+m; n++) {
				signals[i] += lastOutputActivations.get(activationCounter)[n-m] * weightMatrix.weights[i * (multiplierSize) + n + 1];
			}
		}
		return signals;
	}

	@Override
	public double[] errorGradient(double[] eg, double[] input, double[] na) {
		// TODO Auto-generated method stub
		return null;
	}

}
