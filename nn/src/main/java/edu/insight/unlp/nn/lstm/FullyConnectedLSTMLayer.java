package edu.insight.unlp.nn.lstm;

import java.util.HashMap;
import java.util.Map;

import edu.insight.unlp.nn.ActivationFunction;
import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.NNLayer;
import edu.insight.unlp.nn.af.Sigmoid;
import edu.insight.unlp.nn.af.Tanh;
import edu.insight.unlp.nn.common.WeightInitializer;

public class FullyConnectedLSTMLayer extends NNLayer {

	private ActivationFunction afInputGate = new Sigmoid();
	private ActivationFunction afForgetGate = new Sigmoid();
	private ActivationFunction afOutputGate = new Sigmoid();
	private ActivationFunction afCellOutput = new Tanh();
	
	private double[] inputGateWeights, inputGateDeltas;
	private double[] forgetGateWeights, forgetGateDeltas;
	private double[] outputGateWeights, outputGateDeltas;
	
	private Map<Integer, double[]> cellStateLastActivations;
	
	public FullyConnectedLSTMLayer(int numUnits, ActivationFunction af, NN nn) {
		this.numUnits = numUnits;
		this.af = af;
		this.nn = nn;
	}

	public FullyConnectedLSTMLayer(int numUnits, NN nn) {
		new FullyConnectedLSTMLayer(numUnits, new Tanh(), nn);
	}
	
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
		//http://colah.github.io/posts/2015-08-Understanding-LSTMs/
		//forget gate
		double forgetGateSignals[] = computeSignals(input, forgetGateWeights, lastActivations);
		double[] forgetGateActivations = afForgetGate.activation(forgetGateSignals);

		//input gate
		double inputGateSignals[] = computeSignals(input, inputGateWeights, lastActivations);
		double[] inputGateActivations = afInputGate.activation(inputGateSignals);

		// C̃t (see Colah blog), tanh layer, default value of af is tanh, this is the computation of new candidate vector C̃t to be added to the cell state. 
		double[] cellStateInputSignals = computeSignals(input, weights, lastActivations);  
		double[] cellStateInputActivations = af.activation(cellStateInputSignals);
		
		//It’s now time to update the old cell state, Ct−1, into the new cell state Ct
		double[] lastCellStateActivation = cellStateLastActivations.get(activationCounter);
		double[] notToForget = elementMul(forgetGateActivations, lastCellStateActivation);
		double[] cellInput = elementMul(inputGateActivations, cellStateInputActivations);	
		double[] cellStateActivations = new double[numUnits];
		for(int i=0; i<numUnits; i++){
			cellStateActivations[i] = notToForget[i] + cellInput[i];
		}

		//output gate
		double outputGateSignals[] = computeSignals(input, outputGateWeights, lastActivations);  //lastActivations stores the step outputs of the lstm block 
		double[] outputGateActivations = afOutputGate.activation(outputGateSignals);

		//computing lstm block output
		double[] cellStateSquashing = afCellOutput.activation(cellStateActivations);
		double[] output = elementMul(outputGateActivations, cellStateSquashing);

		lastActivations.put(activationCounter, output);
		cellStateLastActivations.put(activationCounter, cellStateActivations);
		return output;
	}

	@Override
	public void initializeLayer(int previousLayerUnits) {
		super.initializeLayer(previousLayerUnits, true);	
		
		int totalWeightParams = (prevLayerUnits+1+numUnits) * numUnits;

		cellStateLastActivations = new HashMap<Integer, double[]>();
		cellStateLastActivations.put(-1, new double[numUnits]);

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
