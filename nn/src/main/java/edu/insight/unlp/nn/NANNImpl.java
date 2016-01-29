package edu.insight.unlp.nn;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.insight.unlp.nn.common.NASequence;
import edu.insight.unlp.nn.common.Sequence;

public class NANNImpl implements NANN, Serializable {

	private static final long serialVersionUID = 1L;
	protected List<NNLayer> layers;
	protected ErrorFunction ef;
	protected NANN naNN;
	public double totalLoss = 0.0;
	public double totalSteps = 0.0;

	private double[][] eg;

	public NANNImpl(ErrorFunction ef, NANN naNN){
		this.ef = ef;
		this.naNN = naNN;
	}

	public NANNImpl(ErrorFunction ef){
		this.ef = ef;
		//this.naNN = naNN;
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
		double finalNaLoss = 0.0;

		for(Sequence seq : training){
			ff(seq, ef, true);
			double[][] naErrors = bp(eg);
			//if(randomBoolean()){
			finalNaLoss = naNN.bpNA(naErrors, seq, learningRate);
			//}
			naNN.resetActivationCounter(true);
			update(learningRate);
			resetActivationCounter(true);
		}
		System.out.println("epoch[ below"+"/" + "below" + "] train loss = " + finalNaLoss);
		return totalLoss/totalSteps;
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

	public static double[] multiplyWeight(double[] array, double weight) {
		int length = array.length;
		double[] result = new double[length];
		int pos = 0;	
		for (double element : array) {
			result[pos] = element * weight;
			pos++;
		}
		return result;
	}

	@Override
	public double[][] ff(Sequence seq, ErrorFunction ef, boolean applyTraining) {
		int i = 0;
		double[][] target = new double[seq.target.length][];
		if(applyTraining)
			eg = new double[seq.target.length][];
		double[][] naInput = ((NASequence) seq).naInput;
		for(int jC=-1; jC<naInput.length-1; jC++){
			double[] naIn = new double[naInput[0].length];
			if(jC!=-1){
				naIn = naInput[jC];
			} 
			double[] overallAnnotationVector = new double[seq.inputSeq[0].length];
			for(double[] input : seq.inputSeq){
				double[] inputToNa = concat(input, naIn);
				double[] naOutput = naNN.ff(inputToNa, applyTraining);
				double naWeight = naOutput[0];
				double[] finalInput = multiplyWeight(input, naWeight);
				int addCounter = 0;
				for(double in : finalInput){
					overallAnnotationVector[addCounter] = overallAnnotationVector[addCounter] + in;
					addCounter++;
				}
			}
			double[] activations = null;
			activations = overallAnnotationVector;		
			for(NNLayer layer : layers){
				activations = layer.computeActivations(activations, applyTraining);
			}
			double[] errors = ef.error(seq.target[i], activations);
			if(applyTraining)
				eg[i] = errors;
			totalLoss = totalLoss + errors[errors.length-1];
			totalSteps++;
			target[i] = activations;
			i++;
		}
		return target;
	}

	public Map<NNLayer, double[][]> ff(Sequence seq, ErrorFunction ef, boolean applyTraining, Set<NNLayer> layersForOutput) {
		Map<NNLayer, double[][]> activationsMap = new HashMap<NNLayer, double[][]>();
		int i = 0;
		//double[][] target = new double[seq.inputSeq.length][];
		for(NNLayer layer : layersForOutput){
			double[][] layerTarget = new double[seq.inputSeq.length][];
			activationsMap.put(layer, layerTarget);
		}
		if(applyTraining)
			eg = new double[seq.inputSeq.length][];
		for(double[] input : seq.inputSeq){
			double[] activations = null;
			activations = input;		
			for(NNLayer layer : layers){
				activations = layer.computeActivations(activations, applyTraining);
				if(activationsMap.containsKey(layer)){
					double[][] layerActivations = activationsMap.get(layer);
					layerActivations[i] = activations;
				}
			}
			double[] errors = ef.error(seq.target[i], activations);
			if(applyTraining)
				eg[i] = errors;
			totalLoss = totalLoss + errors[errors.length-1];
			totalSteps++;
			i++;
		}
		return activationsMap;
	}

	private double[][] bp(double[][] errorGradient){
		int activationCounterVal = layers.get(layers.size()-1).getActivationCounterVal();
		double[][] naErrors = new double[activationCounterVal+1][];
		int naErrorCounter = 0;
		int o = errorGradient[0].length-1;
		double[] stageErrorGradient = null;
		double[] outputFromNextStageEg = new double[o];
		for(int j=layers.get(layers.size()-1).getActivationCounterVal(); j>=0; j--){
			stageErrorGradient = errorGradient[j];
			int rCounter = 0;
			for(double r : outputFromNextStageEg){
				stageErrorGradient[rCounter] = stageErrorGradient[rCounter] + r;
				rCounter++;
			}
			for(int i = layers.size() - 1; i>=0; i--){
				stageErrorGradient = layers.get(i).errorGradient(stageErrorGradient);
				int prevLayerUnits = layers.get(i).prevLayerUnits;
				if(prevLayerUnits!=-1){
					if(stageErrorGradient.length > prevLayerUnits+1){
						double[] nextOutputStageError = Arrays.copyOfRange(stageErrorGradient, prevLayerUnits+1, stageErrorGradient.length-1);
						int outputUnitCounter = 0;
						for(double oD : nextOutputStageError){
							outputFromNextStageEg[outputUnitCounter] = outputFromNextStageEg[outputUnitCounter] + oD;
							outputUnitCounter++;
						}
						stageErrorGradient = Arrays.copyOfRange(stageErrorGradient, 0, prevLayerUnits+1);
					}
				}
			}
			naErrors[naErrorCounter++] = stageErrorGradient;
			double totalError = stageErrorGradient[stageErrorGradient.length-1];
			if(j-1>-1){
				errorGradient[j-1][o] = totalError;
			}
		}
		return naErrors;
	}

	public void resetActivationCounter(boolean training){
		for(NNLayer layer : layers){
			layer.resetActivationCounter(training);
		}
	}

	public void cleanUpTheMess(){
		for(NNLayer layer : layers){
			layer.cleanUpTheMess();
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

	public void resetError(){
		totalLoss = 0.0;
		totalSteps = 0.0;
	}

	@Override
	public double getError() {
		return totalLoss/totalSteps;
	}

	public boolean randomBoolean(){
		return Math.random() < 0.5;
	}

	public double bpNA(double[][] errorGradients, Sequence seq, double learningRate){
		double[][] inputSeq = seq.inputSeq;
		double[][] naInput = ((NASequence) seq).naInput;
		double totalLoss = 0.0;
		double totalSteps = 0.0;
		for(int egC=0; egC<errorGradients.length; egC++){
			double[] eg = errorGradients[egC];
			double[] ds = new double[naInput[0].length];
			if(egC != 0){
				ds = naInput[errorGradients.length-1-egC];	
			}
			int iC = inputSeq.length;
			//for(double[] inp : inputSeq){
			for(iC=inputSeq.length-1; iC>=0; iC--){
				double[] inp = inputSeq[iC];
				double[] elementMulRes = elementMul(eg, inp);
				double finalEg  = 0.0;
				double[][] finalEgs = new double[1][2];
				for(double el : elementMulRes){
					finalEg = finalEg + el;
				}
				finalEgs[0][0] = finalEg;
				finalEgs[0][1] = finalEg;
				totalLoss = totalLoss + finalEg;
				totalSteps++;
				bpNA(finalEgs, inp, ds);
				update(learningRate);
			}
		}
		resetActivationCounter(true);
		return totalLoss/totalSteps;
	}

	public void bpNA(double[][] errorGradients, double[] inp, double[] na){
		int o = errorGradients[0].length-1;
		double[] stageErrorGradient = null;
		//for(int j=layers.get(layers.size()-1).getActivationCounterVal(); j>=0; j--){
		for(int j=0; j>=0; j--){
			stageErrorGradient = errorGradients[j];
			for(int i = layers.size() - 1; i>=0; i--){
				stageErrorGradient = layers.get(i).errorGradient(stageErrorGradient, inp, na);
			}
			double totalError = stageErrorGradient[stageErrorGradient.length-1];
			if(j-1>-1){
				errorGradients[j-1][o] = totalError;
			}
		}
	}

	public double[] ff(double[] input, boolean applyTraining) {
		double[] activations = null;
		activations = input;		
		for(NNLayer layer : layers){
			activations = layer.computeActivations(activations, applyTraining);
		}
		return activations;
	}

	private double[] elementMul(double[] one, double[] two){
		int numUnits = two.length; 
		double[] result = new double[numUnits];
		for(int i=0; i<numUnits; i++){
			result[i] = one[i] * two[i]; 
		}
		return result;
	}


}
