package edu.insight.unlp.nn.lstm;

import java.util.HashMap;
import java.util.Map;

import edu.insight.unlp.nn.ActivationFunction;
import edu.insight.unlp.nn.NNLayer;
import edu.insight.unlp.nn.af.Sigmoid;
import edu.insight.unlp.nn.common.WeightInitializer;

public class FullyConnectedLSTMLayer extends NNLayer {

	private ActivationFunction afInputGate = new Sigmoid();
	private ActivationFunction afForgetGate = new Sigmoid();
	private ActivationFunction afOutputGate = new Sigmoid();

	private double[] inputGateWeights, inputGateDeltas;
	private double[] forgetGateWeights, forgetGateDeltas;
	private double[] outputGateWeights, outputGateDeltas;

	private Map<Integer, double[]> contextLastActivations;

	//ActivationFunction fCellInput = new TanhUnit();
	//ActivationFunction fCellOutput = new TanhUnit();

	@Override
	public double[] errorGradient(double[] input) {
		return null;
	}

	private double[] computeSignals(double[] input, double[] weights, Map<Integer, double[]> activations) {
		double signals[] = new double[numUnits];
		for (int i = 0; i < signals.length; i++) {
			signals[i] = 1 * weights[i * (input.length + 1 + numUnits)]; //the bias one, multiplied the weight by 1, so added directly to outputs
			int j = 0;
			for (j = 0; j < input.length; j++) {
				signals[i] += input[j] * weights[i * (input.length + 1 + numUnits) + j + 1];
			}
			for (int m = j; m < numUnits+j; m++) {
				signals[i] += activations.get(activationCounter)[m-j] * weights[i * (input.length + 1 + numUnits) + m + 1];
			}
		}
		return signals;
	}

	private double[] elementMul(double[] one, double[] two){
		double[] result = new double[numUnits];
		for(int i=0; i<numUnits; i++){
			result[i] = one[i] * two[i]; 
		}
		return result;
	}

	@Override
	public double[] computeSignals(double[] input) {
		//input gate
		double inputGateSignals[] = computeSignals(input, inputGateWeights, contextLastActivations);
		double[] inputGateActivations = afInputGate.activation(inputGateSignals);

		//forget gate
		double forgetGateSignals[] = computeSignals(input, forgetGateWeights, contextLastActivations);
		double[] forgetGateActivations = afForgetGate.activation(forgetGateSignals);

		//output gate
		double outputGateSignals[] = computeSignals(input, outputGateWeights, contextLastActivations);
		double[] outputGateActivations = afOutputGate.activation(outputGateSignals);

		//write operation on cells
		double[] cellInputSignals = computeSignals(input, weights, contextLastActivations);
		double[] cellInputActivations = af.activation(cellInputSignals);

		//compute new cell activation
		double[] lastActivation = lastActivations.get(activationCounter);

		double[] retainCell = elementMul(forgetGateActivations, lastActivation);
		double[] writeCell = elementMul(inputGateActivations, cellInputActivations);	

		double[] activations = new double[numUnits];
		for(int i=0; i<numUnits; i++){
			activations[i] = retainCell[i] + writeCell[i];
		}

		//compute hidden state as gated, saturated cell activations
		double[] outputSquash = afOutputGate.activation(activations);
		double[] output = elementMul(outputGateActivations, outputSquash);

		//rollover activations for next iteration
		lastActivations.put(activationCounter, activations);
		contextLastActivations.put(activationCounter, output);
		return output;
	}

	@Override
	public void initializeLayer(int previousLayerUnits) {
		super.initializeLayer(previousLayerUnits, true);	

		int totalWeightParams = (prevLayerUnits+1+numUnits) * numUnits;

		contextLastActivations = new HashMap<Integer, double[]>();
		contextLastActivations.put(-1, new double[numUnits]);

		inputGateWeights = new double[totalWeightParams];
		WeightInitializer.randomInitializeLeCun(inputGateWeights);//(weights, 0.2);
		inputGateDeltas = new double[inputGateWeights.length];

		outputGateWeights = new double[totalWeightParams];
		WeightInitializer.randomInitializeLeCun(outputGateWeights);//(weights, 0.2);
		outputGateDeltas = new double[outputGateWeights.length];

		forgetGateWeights = new double[totalWeightParams];
		WeightInitializer.randomInitializeLeCun(forgetGateWeights);//(weights, 0.2);
		forgetGateDeltas = new double[forgetGateWeights.length];

	}
}
