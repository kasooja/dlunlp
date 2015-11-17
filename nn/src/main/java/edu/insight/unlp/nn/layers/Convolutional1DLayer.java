package edu.insight.unlp.nn.layers;

import java.util.HashMap;
import java.util.Map;

import edu.insight.unlp.nn.ActivationFunction;
import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.NNLayer;
import edu.insight.unlp.nn.common.WeightMatrix;

public class Convolutional1DLayer extends NNLayer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private int[] k; //kernels
	//private int[] kernelNumUnits; //numUnits for specific kernels
	private WeightMatrix[] kernelWeightMatrices;
	private int v; //vectorLength

	public Convolutional1DLayer(ActivationFunction af, NN nn, int[] k, int v){
		//this.kernelNumUnits = kernelNumUnits;
		this.af = af;
		this.nn = nn;
		this.k = k;
		this.v = v;
		kernelWeightMatrices = new WeightMatrix[this.k.length];
	}

	@Override
	public double[] errorGradient(double[] input) {
		return null;
	}

	public double[] computeSignals(double[] input, WeightMatrix weightMatrix, Map<Integer, double[]> activations) {  //MLP does not require its activations for feedback, so not used in the method for computing signals
		if(prevLayerUnits == -1){
			return input;
		}

		for(int j = 0; j<kernelWeightMatrices.length; j++){
			WeightMatrix kernelWeightMatrix = kernelWeightMatrices[j];
			int kernelSize = k[j];
			int inputSplit = kernelSize * v;
			int prevLayerAbstractUnits = prevLayerUnits / v;
			int kernelFeatureMapOutputSize = prevLayerAbstractUnits - kernelSize + 1;
			double featureMap[] = new double[kernelFeatureMapOutputSize];
			
//			for(int i = 0; i < outputs.length; i++) {
//				outputs[i] = 1 * kernelWeightMatrix.weights[i * (input.length + 1)]; //the bias one, multiplied the weight by 1, so added directly to outputs
//				for (int m = 0; m < input.length; m++) {
//					outputs[i] += input[m] * kernelWeightMatrix.weights[i * (input.length + 1) + m + 1];
//				}
//			}
		}
		//return outputs;
		return null;
	}

	@Override
	public void initializeLayer(int previousLayerUnits) {
		this.prevLayerUnits = previousLayerUnits;
		lastActivations = new HashMap<Integer, double[]>();
		lastActivations.put(-1, new double[numUnits]);
		if(prevLayerUnits==-1){
			return;
		}
		lastActivationDerivatives = new HashMap<Integer, double[]>();
		for(int i=0; i<k.length; i++){
			int totalWeightParams = 1 * (k[i] * v + 1);
			kernelWeightMatrices[i].weights = new double[1 * (k[i] * v + 1)];
			kernelWeightMatrices[i].biasMultiplier = (k[i] * v + 1);
			kernelWeightMatrices[i].deltas = new double[totalWeightParams];
			kernelWeightMatrices[i].stepCache = new double[totalWeightParams];
			initializeLayerWeights(kernelWeightMatrices[i], 0.0);
		}
	}

}
