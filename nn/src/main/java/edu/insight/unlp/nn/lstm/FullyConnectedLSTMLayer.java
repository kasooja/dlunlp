package edu.insight.unlp.nn.lstm;

import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

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

	private double[] inputGateWeights, inputGateDeltas, inputGateStepCache;
	private double[] forgetGateWeights, forgetGateDeltas, forgetGateStepCache;
	private double[] outputGateWeights, outputGateDeltas, outputGateStepCache;

	private Map<Integer, double[]> lastOutputGateDerivatives;
	private Map<Integer, double[]> lastForgetGateDerivatives;
	private Map<Integer, double[]> lastInputGateDerivatives;

	private Map<Integer, double[]> lastOutputGateActivations;
	private Map<Integer, double[]> lastInputGateActivations;
	private Map<Integer, double[]> cellStateLastActivations;
	private Map<Integer, double[]> lastCellStateInputActivations;

	private double[] nextStageError;

	public FullyConnectedLSTMLayer(int numUnits, ActivationFunction af, NN nn) {
		this.numUnits = numUnits;
		this.af = af;
		this.nn = nn;
	}

	public FullyConnectedLSTMLayer(int numUnits, NN nn) {
		this(numUnits, new Tanh(), nn);
	}

	@Override
	public double[] errorGradient(double[] eg) {
		for(int i=0; i<eg.length-1; i++){
			eg[i] = eg[i] + nextStageError[i];
		}
		int currentIndex = nn.getLayers().indexOf(this);
		if(currentIndex!=0){
			NNLayer prevLayer = nn.getLayers().get(currentIndex-1);

			double[] lastStageOutput = lastActivations.get(activationCounter);

			double[] derivatives = lastActivationDerivatives.get(activationCounter);

			double[] cellStateActivations = cellStateLastActivations.get(activationCounter);
			double[] cellStateSquashing = afCellOutput.activation(cellStateActivations);
			double[] cellStateDerivatives = afCellOutput.activationDerivative(cellStateActivations);

			double[] outputGateActivation = lastOutputGateActivations.get(activationCounter);
			double[] forgetGateDerivatives = lastForgetGateDerivatives.get(activationCounter);
			double[] inputGateDerivatives = lastInputGateDerivatives.get(activationCounter);

			Map<Integer, double[]> prevLayerActivationsMap = prevLayer.lastActivations();
			double[] prevLayerActivations = prevLayerActivationsMap.get(activationCounter);
			double[] outputGateDerivatives = lastOutputGateDerivatives.get(activationCounter);

			double[] egOutputGatePrevLayer = new double[prevLayerUnits + 1];
			double[] egOutputGatePrevStage = new double[numUnits + 1];

			double[] egForgetGatePrevLayer = new double[prevLayerUnits + 1];
			double[] egForgetGatePrevStage = new double[numUnits + 1];

			double[] egInputGatePrevLayer = new double[prevLayerUnits + 1];
			double[] egInputGatePrevStage = new double[numUnits + 1];

			double[] egCellStateInputPrevLayer = new double[prevLayerUnits + 1];
			double[] egCellStateInputPrevStage = new double[numUnits + 1];

			for(int i=0; i<eg.length-1; i++){
				int currentWeightIndex = i * (1 + prevLayerUnits + numUnits);			
				double lambda = eg[i] * outputGateDerivatives[i] * cellStateSquashing[i];
				outputGateDeltas[currentWeightIndex] = outputGateDeltas[currentWeightIndex] +  1 * lambda; //the bias one, multiplied the weight by 1, so added directly to outputs
				int j = 0;
				for(j=0; j<prevLayerUnits; j++){					
					double delta = lambda * prevLayerActivations[j];
					outputGateDeltas[currentWeightIndex + j + 1] = outputGateDeltas[currentWeightIndex + j + 1] +  delta;
					egOutputGatePrevLayer[j] = egOutputGatePrevLayer[j] + delta * outputGateWeights[currentWeightIndex + j + 1];
				}
				for(int m=j; m<numUnits+j; m++){					
					double delta = lambda * lastStageOutput[m-j];
					outputGateDeltas[currentWeightIndex + m + 1] = outputGateDeltas[currentWeightIndex + m + 1] + delta;
					egOutputGatePrevStage[m-j] = egOutputGatePrevStage[m-j] + delta * outputGateWeights[currentWeightIndex + m + 1];
				}
			}

			for(int i=0; i<eg.length-1; i++){
				int currentWeightIndex = i * (1 + prevLayerUnits + numUnits);			
				double lambda = eg[i] * forgetGateDerivatives[i] * cellStateDerivatives[i] * outputGateActivation[i] * cellStateActivations[i];
				forgetGateDeltas[currentWeightIndex] = forgetGateDeltas[currentWeightIndex] +  1 * lambda; //the bias one, multiplied the weight by 1, so added directly to outputs
				int j = 0;
				for(j=0; j<prevLayerUnits; j++){					
					double delta = lambda * prevLayerActivations[j];
					forgetGateDeltas[currentWeightIndex + j + 1] = forgetGateDeltas[currentWeightIndex + j + 1] +  delta;
					egForgetGatePrevLayer[j] = egForgetGatePrevLayer[j] + delta * forgetGateWeights[currentWeightIndex + j + 1];
				}
				for(int m=j; m<numUnits+j; m++){					
					double delta = lambda * lastStageOutput[m-j];
					forgetGateDeltas[currentWeightIndex + m + 1] = forgetGateDeltas[currentWeightIndex + m + 1] + delta;
					egForgetGatePrevStage[m-j] = egForgetGatePrevStage[m-j] + delta * forgetGateWeights[currentWeightIndex + m + 1];
				}
			}

			for(int i=0; i<eg.length-1; i++){
				int currentWeightIndex = i * (1 + prevLayerUnits + numUnits);			
				double lambda = eg[i] * derivatives[i] * cellStateDerivatives[i] * outputGateActivation[i] * lastInputGateActivations.get(activationCounter)[i] ;
				deltas[currentWeightIndex] = deltas[currentWeightIndex] +  1 * lambda; //the bias one, multiplied the weight by 1, so added directly to outputs
				int j = 0;
				for(j=0; j<prevLayerUnits; j++){					
					double delta = lambda * prevLayerActivations[j];
					deltas[currentWeightIndex + j + 1] = deltas[currentWeightIndex + j + 1] +  delta;
					egCellStateInputPrevLayer[j] = egCellStateInputPrevLayer[j] + delta * weights[currentWeightIndex + j + 1];
				}
				for(int m=j; m<numUnits+j; m++){					
					double delta = lambda * lastStageOutput[m-j];
					deltas[currentWeightIndex + m + 1] = deltas[currentWeightIndex + m + 1] + delta;
					egCellStateInputPrevStage[m-j] = egCellStateInputPrevStage[m-j] + delta * weights[currentWeightIndex + m + 1];
				}
			}

			for(int i=0; i<eg.length-1; i++){
				int currentWeightIndex = i * (1 + prevLayerUnits + numUnits);			
				double lambda = eg[i] * inputGateDerivatives[i] * cellStateDerivatives[i] * outputGateActivation[i] * lastCellStateInputActivations.get(activationCounter)[i] ;
				inputGateDeltas[currentWeightIndex] = inputGateDeltas[currentWeightIndex] +  1 * lambda; //the bias one, multiplied the weight by 1, so added directly to outputs
				int j = 0;
				for(j=0; j<prevLayerUnits; j++){					
					double delta = lambda * prevLayerActivations[j];
					inputGateDeltas[currentWeightIndex + j + 1] = inputGateDeltas[currentWeightIndex + j + 1] +  delta;
					egInputGatePrevLayer[j] = egInputGatePrevLayer[j] + delta * inputGateWeights[currentWeightIndex + j + 1];
				}
				for(int m=j; m<numUnits+j; m++){					
					double delta = lambda * lastStageOutput[m-j];
					inputGateDeltas[currentWeightIndex + m + 1] = inputGateDeltas[currentWeightIndex + m + 1] + delta;
					egInputGatePrevStage[m-j] = egInputGatePrevStage[m-j] + delta * inputGateWeights[currentWeightIndex + m + 1];
				}
			}

			double[] finalEgPrevLayer = new double[prevLayerUnits + 1];
			double[] finalEgPrevStage = new double[numUnits + 1];

			for(int i=0; i<finalEgPrevLayer.length; i++){
				finalEgPrevLayer[i] = egCellStateInputPrevLayer[i] + egForgetGatePrevLayer[i] + 
						egInputGatePrevLayer[i] + egOutputGatePrevLayer[i];
			}

			for(int i=0; i<finalEgPrevStage.length; i++){
				finalEgPrevStage[i] = egCellStateInputPrevStage[i] + egForgetGatePrevStage[i] + 
						egInputGatePrevStage[i] + egOutputGatePrevStage[i];
			}

			finalEgPrevLayer[prevLayerUnits] = eg[eg.length-1];
			finalEgPrevStage[numUnits] = eg[eg.length-1];

			//lastActivations.put(activationCounter, null);
			//lastActivationDerivatives.put(activationCounter, null);
			nextStageError = finalEgPrevStage;
			activationCounter--;
			return finalEgPrevLayer;
		}
		return null;
	}

	public double[] computeSignals(double[] input, double[] weights, Map<Integer, double[]> activations) {
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

	@Override
	public double[] computeActivations(double[] input, boolean training) {
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

			//storing intGateActivations
			lastInputGateActivations.put(activationCounter, inputGateActivations);

			//storing cellInputActivations
			lastCellStateInputActivations.put(activationCounter, cellStateInputActivations);

		}
		return output;
	}

	@Override
	public void initializeLayer(int previousLayerUnits) {
		super.initializeLayer(previousLayerUnits, true);	

		nextStageError = new double[numUnits + 1];

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

		lastOutputGateDerivatives = new HashMap<Integer, double[]>();
		lastOutputGateDerivatives.put(-1, new double[numUnits]);

		lastForgetGateDerivatives = new HashMap<Integer, double[]>();
		lastForgetGateDerivatives.put(-1, new double[numUnits]);

		lastInputGateDerivatives = new HashMap<Integer, double[]>();
		lastInputGateDerivatives.put(-1, new double[numUnits]);

		lastOutputGateActivations = new HashMap<Integer, double[]>();
		lastOutputGateActivations.put(-1, new double[numUnits]);

		lastInputGateActivations = new HashMap<Integer, double[]>();
		lastInputGateActivations.put(-1, new double[numUnits]);

		cellStateLastActivations = new HashMap<Integer, double[]>();
		cellStateLastActivations.put(-1, new double[numUnits]);

		lastCellStateInputActivations = new HashMap<Integer, double[]>();
		lastCellStateInputActivations.put(-1, new double[numUnits]);


	}

	public void resetActivationCounter(boolean training){
		super.resetActivationCounter(training);
		if(training && prevLayerUnits!=-1){
			inputGateDeltas = new double[inputGateWeights.length];
			inputGateStepCache = new double[inputGateWeights.length];

			forgetGateDeltas = new double[forgetGateWeights.length];
			forgetGateStepCache = new double[forgetGateWeights.length];

			outputGateDeltas = new double[outputGateWeights.length];
			outputGateStepCache = new double[outputGateWeights.length];
		}
		if(training)
			nextStageError = new double[numUnits + 1];
	}

	public void update(double learningRate) {
		super.update(learningRate, forgetGateWeights, forgetGateDeltas, forgetGateStepCache);
		super.update(learningRate, inputGateWeights, inputGateDeltas, inputGateStepCache);
		super.update(learningRate, weights, deltas, stepCache);
		super.update(learningRate, outputGateWeights, outputGateDeltas, outputGateStepCache);
	}

	private double[] elementMul(double[] one, double[] two){
		double[] result = new double[numUnits];
		for(int i=0; i<numUnits; i++){
			result[i] = one[i] * two[i]; 
		}
		return result;
	}

}
