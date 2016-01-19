package edu.insight.unlp.nn.layers;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

import edu.insight.unlp.nn.ActivationFunction;
import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.NNLayer;
import edu.insight.unlp.nn.af.Sigmoid;
import edu.insight.unlp.nn.af.Tanh;
import edu.insight.unlp.nn.common.WeightMatrix;

public class FullyConnectedLSTMLayer extends NNLayer {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private ActivationFunction afInputGate, afForgetGate, afOutputGate, afCellOutput;

	private WeightMatrix inputGateMatrix;
	private WeightMatrix forgetGateMatrix;
	private WeightMatrix outputGateMatrix;

	private Map<Integer, double[]> lastOutputGateDerivatives,  lastForgetGateDerivatives, lastInputGateDerivatives;

	private Map<Integer, double[]> lastOutputGateActivations, lastInputGateActivations, lastForgetGateActivations;

	private Map<Integer, double[]> cellStateLastActivations; //rename 
	private Map<Integer, double[]> lastCellStateInputActivations; //rename

	private double[] nextStageOutputError, nextStageCellStateError;

	public FullyConnectedLSTMLayer(int numUnits, ActivationFunction af, NN nn) {
		this.numUnits = numUnits;
		this.af = af;
		this.nn = nn;
		afInputGate = new Sigmoid();
		afForgetGate = new Sigmoid();
		afOutputGate = new Sigmoid();
		afCellOutput = new Tanh();

		weightMatrix = new WeightMatrix();
		inputGateMatrix = new WeightMatrix();
		forgetGateMatrix = new WeightMatrix();
		outputGateMatrix = new WeightMatrix();

	}

	public FullyConnectedLSTMLayer(int numUnits, NN nn) {
		this(numUnits, new Tanh(), nn);
	}

	public double[] errorGradient(double[] eg, double[] lambda, double[] prevLayerActivations, double[] feedback, WeightMatrix weightMatrix){
		double[] egPrev = new double[prevLayerUnits + numUnits];
		for(int i=0; i<eg.length-1; i++){
			int currentWeightIndex = i * (1 + prevLayerUnits + numUnits);			
			weightMatrix.deltas[currentWeightIndex] = weightMatrix.deltas[currentWeightIndex] +  1 * lambda[i]; //the bias one, multiplied the weight by 1, so added directly to outputs
			int j = 0;
			for(j=0; j<prevLayerUnits; j++){					
				double delta = lambda[i] * prevLayerActivations[j];
				weightMatrix.deltas[currentWeightIndex + j + 1] = weightMatrix.deltas[currentWeightIndex + j + 1] +  delta;
				egPrev[j] = egPrev[j] + lambda[i] * weightMatrix.weights[currentWeightIndex + j + 1];
			}
			for(int m=j; m<numUnits+j; m++){					
				double delta = lambda[i] * feedback[m-j];
				weightMatrix.deltas[currentWeightIndex + m + 1] = weightMatrix.deltas[currentWeightIndex + m + 1] + delta;
				egPrev[prevLayerUnits + m-j] = egPrev[prevLayerUnits + m-j] + lambda[i] * weightMatrix.weights[currentWeightIndex + m + 1];
			}
		}
		return egPrev;
	}

	public double[] errorGradient(double[] eg) {
		for(int i=0; i<eg.length-1; i++){
			eg[i] = eg[i] + nextStageOutputError[i];
		}
		double[] finalEgPrevLayer = eg;
		int currentIndex = nn.getLayers().indexOf(this);
		if(currentIndex!=0){
			NNLayer prevLayer = nn.getLayers().get(currentIndex-1);
			//recomputing it, didnt store, needed for the backprop
			double[] cellStateActivations = cellStateLastActivations.get(activationCounter);
			double[] cellStateSquashing = afCellOutput.activation(cellStateActivations);
			//assuming default value of afCellOutput as tanh, so d/dx tanh = 1 - (Math.pow(tanh, 2));
			double[] cellStateDerivatives = new double[numUnits]; 
			IntStream.range(0, numUnits).forEach(i -> cellStateDerivatives[i] = 1-(Math.pow(cellStateSquashing[i], 2)));		

			double[] outputGateLambda = new double[numUnits];
			double[] forgetGateLambda = new double[numUnits];			
			double[] cellStateInputLambda = new double[numUnits];
			double[] inputGateLambda = new double[numUnits];

			for(int i=0; i<numUnits; i++){
				outputGateLambda[i] = eg[i] * lastOutputGateDerivatives.get(activationCounter)[i] * cellStateSquashing[i];
				forgetGateLambda[i] = (eg[i] * lastOutputGateActivations.get(activationCounter)[i] * cellStateDerivatives[i] + nextStageCellStateError[i]) * 
						cellStateLastActivations.get(activationCounter-1)[i] * lastForgetGateDerivatives.get(activationCounter)[i] ;
				cellStateInputLambda[i] = (eg[i] * lastOutputGateActivations.get(activationCounter)[i] * cellStateDerivatives[i] + nextStageCellStateError[i]) *
						lastInputGateActivations.get(activationCounter)[i] * lastActivationDerivatives.get(activationCounter)[i];
				inputGateLambda[i] = (eg[i] * lastOutputGateActivations.get(activationCounter)[i] * cellStateDerivatives[i] + nextStageCellStateError[i]) * 
						lastCellStateInputActivations.get(activationCounter)[i] * lastInputGateDerivatives.get(activationCounter)[i];
			}

			double[] egOutputGate = errorGradient(eg, outputGateLambda, prevLayer.lastActivations().get(activationCounter), 
					lastActivations.get(activationCounter-1), outputGateMatrix);
			double[] egForgetGate = errorGradient(eg, forgetGateLambda, prevLayer.lastActivations().get(activationCounter), 
					lastActivations.get(activationCounter-1), forgetGateMatrix);
			double[] egCellStateInputGate = errorGradient(eg, cellStateInputLambda, prevLayer.lastActivations().get(activationCounter),
					lastActivations.get(activationCounter-1), weightMatrix);
			double[] egInputGate = errorGradient(eg, inputGateLambda, prevLayer.lastActivations().get(activationCounter), 
					lastActivations.get(activationCounter-1), inputGateMatrix);

			finalEgPrevLayer = new double[prevLayerUnits + 1];
			double[] finalEgPrevStage = new double[numUnits + 1];

			for(int i=0; i<prevLayerUnits; i++){
				finalEgPrevLayer[i] = egOutputGate[i] + egForgetGate[i] + 
						egCellStateInputGate[i] + egInputGate[i];
			}

			for(int i=prevLayerUnits; i<prevLayerUnits + numUnits; i++){
				finalEgPrevStage[i-prevLayerUnits] = egOutputGate[i] + egForgetGate[i] + 
						egCellStateInputGate[i] + egInputGate[i];
			}

			finalEgPrevLayer[prevLayerUnits] = eg[eg.length-1];
			finalEgPrevStage[numUnits] = eg[eg.length-1];
			nextStageOutputError = finalEgPrevStage;

			for(int i=0; i<numUnits; i++){
				nextStageCellStateError[i] = lastForgetGateActivations.get(activationCounter)[i] * (eg[i] * lastOutputGateActivations.get(activationCounter)[i] * cellStateDerivatives[i] + nextStageCellStateError[i]); 
			}
			nextStageCellStateError[numUnits] = eg[eg.length - 1];
		} 
		resetActivationAndDerivatives(activationCounter);
		activationCounter--;
		return finalEgPrevLayer;
	}

	private void resetActivationAndDerivatives(int activationCounter){
		lastActivations.put(activationCounter, null);
		lastActivationDerivatives.put(activationCounter, null);

		lastOutputGateDerivatives.put(activationCounter,  null); 
		lastForgetGateDerivatives.put(activationCounter,  null); 
		lastInputGateDerivatives.put(activationCounter,  null);

		lastOutputGateActivations.put(activationCounter,  null); 
		lastInputGateActivations.put(activationCounter,  null);
		lastForgetGateActivations.put(activationCounter,  null);

		cellStateLastActivations.put(activationCounter,  null); 
		lastCellStateInputActivations.put(activationCounter,  null);
	}


	public double[] computeSignals(double[] input, WeightMatrix weightMatrix, Map<Integer, double[]> activations) {
		double signals[] = new double[numUnits];
		for (int i = 0; i < signals.length; i++) {
			signals[i] = 1 * weightMatrix.weights[i * (input.length + 1 + numUnits)]; //the bias one, multiplied the weight by 1, so added directly to outputs
			int j = 0;
			for (j = 0; j < input.length; j++) {
				signals[i] += input[j] * weightMatrix.weights[i * (input.length + 1 + numUnits) + j + 1];
			}
			for (int m = j; m < numUnits+j; m++) {
				signals[i] += activations.get(activationCounter)[m-j] * weightMatrix.weights[i * (input.length + 1 + numUnits) + m + 1];
			}
		}
		return signals;
	}

	@Override
	public double[] computeActivations(double[] input, boolean training) {
		//http://colah.github.io/posts/2015-08-Understanding-LSTMs/
		//forget gate
		double forgetGateSignals[] = computeSignals(input, forgetGateMatrix, lastActivations);
		double[] forgetGateActivations = afForgetGate.activation(forgetGateSignals);

		//input gate
		double inputGateSignals[] = computeSignals(input, inputGateMatrix, lastActivations);
		double[] inputGateActivations = afInputGate.activation(inputGateSignals);

		// C̃t (see Colah blog), tanh layer, default value of af is tanh, this is the computation of new candidate vector C̃t to be added to the cell state. 
		double[] cellStateInputSignals = computeSignals(input, weightMatrix, lastActivations);  
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
		double outputGateSignals[] = computeSignals(input, outputGateMatrix, lastActivations);  //lastActivations stores the step outputs of the lstm block 
		double[] outputGateActivations = afOutputGate.activation(outputGateSignals);

		//computing lstm block output
		double[] cellStateSquashing = afCellOutput.activation(cellStateActivations);//last tanh on cell state to put the values in -1to1 range
		double[] output = elementMul(outputGateActivations, cellStateSquashing);

		activationCounter++;
		lastActivations.put(activationCounter, output);
		cellStateLastActivations.put(activationCounter, cellStateActivations);

		if(training && prevLayerUnits != -1){
			//for storing derivatives, if required
			double[] outputGateDerivatives = new double[outputGateActivations.length];
			IntStream.range(0, outputGateSignals.length).forEach(i -> outputGateDerivatives[i] = afOutputGate.activationDerivative(outputGateSignals[i]));
			lastOutputGateDerivatives.put(activationCounter, outputGateDerivatives);

			double[] forgetGateDerivatives = new double[forgetGateActivations.length];
			IntStream.range(0, forgetGateSignals.length).forEach(i -> forgetGateDerivatives[i] = afForgetGate.activationDerivative(forgetGateSignals[i]));
			lastForgetGateDerivatives.put(activationCounter, forgetGateDerivatives);

			double[] cellStateInputDerivatives = new double[cellStateInputActivations.length];
			IntStream.range(0, cellStateInputSignals.length).forEach(i -> cellStateInputDerivatives[i] = af.activationDerivative(cellStateInputSignals[i]));
			lastActivationDerivatives.put(activationCounter, cellStateInputDerivatives);

			double[] inputGateDerivatives = new double[inputGateActivations.length];
			IntStream.range(0, inputGateSignals.length).forEach(i -> inputGateDerivatives[i] = afInputGate.activationDerivative(inputGateSignals[i]));
			lastInputGateDerivatives.put(activationCounter, inputGateDerivatives);

			//storing outputGateActivations
			lastOutputGateActivations.put(activationCounter, outputGateActivations);

			//storing inputGateActivations
			lastInputGateActivations.put(activationCounter, inputGateActivations);

			//storing forgetGateActivations
			lastForgetGateActivations.put(activationCounter, forgetGateActivations);

			//storing cellInputActivations
			lastCellStateInputActivations.put(activationCounter, cellStateInputActivations);

		}
		return output;
	}

	@Override
	public void initializeLayer(int previousLayerUnits) {
		super.initializeLayer(previousLayerUnits, true);	

		lastOutputGateActivations = new HashMap<Integer, double[]>();
		lastInputGateActivations = new HashMap<Integer, double[]>();
		lastForgetGateActivations = new HashMap<Integer, double[]>();

		cellStateLastActivations = new HashMap<Integer, double[]>();
		cellStateLastActivations.put(-1, new double[numUnits]);

		lastCellStateInputActivations = new HashMap<Integer, double[]>();

		nextStageOutputError = new double[numUnits + 1];
		nextStageCellStateError = new double[numUnits+1];

		initializeLayerWeights(inputGateMatrix, true, 0.0);
		initializeLayerWeights(outputGateMatrix, true, 0.0);
		initializeLayerWeights(forgetGateMatrix, true, 1.0);

		lastOutputGateDerivatives = new HashMap<Integer, double[]>();
		lastForgetGateDerivatives = new HashMap<Integer, double[]>();
		lastInputGateDerivatives = new HashMap<Integer, double[]>();
	}

	public void resetActivationCounter(boolean training){
		super.resetActivationCounter(training);
		if(training){
			nextStageOutputError = new double[numUnits + 1];
			nextStageCellStateError = new double[numUnits + 1];
		}
	}

	public void update(double learningRate) {
		super.update(learningRate, forgetGateMatrix);
		super.update(learningRate, inputGateMatrix);
		super.update(learningRate, weightMatrix);
		super.update(learningRate, outputGateMatrix);
	}

	private double[] elementMul(double[] one, double[] two){
		double[] result = new double[numUnits];
		for(int i=0; i<numUnits; i++){
			result[i] = one[i] * two[i]; 
		}
		return result;
	}

	@Override
	public double[] errorGradient(double[] eg, double[] input, double[] na) {
		// TODO Auto-generated method stub
		return null;
	}

}
