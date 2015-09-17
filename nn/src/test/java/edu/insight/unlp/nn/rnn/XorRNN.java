
package edu.insight.unlp.nn.rnn;

import java.util.ArrayList;
import java.util.List;

import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.NNLayer;
import edu.insight.unlp.nn.af.Sigmoid;
import edu.insight.unlp.nn.common.SequenceM21;
import edu.insight.unlp.nn.ef.SquareErrorFunction;
import edu.insight.unlp.nn.mlp.InputLayer;
import edu.insight.unlp.nn.rnn.FullyConnectedRNNLayer;
import edu.insight.unlp.nn.rnn.RNN;

public class XorRNN {

	public static double test(NN network, double[][] inputs, double[][] targets) {
		int correct = 0;
		int j = 0;
		double[][] inputSeq = null;
		int counter = 0;
		for(double[] test : inputs) {
			if(j==0){
				inputSeq = new double[2][];
			}
			inputSeq[j] = test;
			j++;
			if(j>=2){
				double[] output = network.outputSequence(inputSeq);
				if(targets[counter][0] == Math.round(output[0])){
					correct++; 
				}
				j=0;
				counter++;
				network.resetActivationCounter();				
			}			
		}
		return ((double)correct)/inputs.length;
	}

	public static void main(String[] args) {
		RNN nn = new RNN(new SquareErrorFunction());
		FullyConnectedRNNLayer outputLayer = new FullyConnectedRNNLayer(1, new Sigmoid(), nn);
		FullyConnectedRNNLayer hiddenLayer = new FullyConnectedRNNLayer(2, new Sigmoid(), nn);
		InputLayer inputLayer = new InputLayer(1);
		List<NNLayer> layers = new ArrayList<NNLayer>();
		layers.add(inputLayer);
		layers.add(hiddenLayer);
		layers.add(outputLayer);
		nn.setLayers(layers);
		nn.initializeNN();

		System.err.print("Reading data...");
		double[][] trainingData = new double[][]{new double[]{0}, new double[]{0}, new double[]{1}, new double[]{1}, new double[]{0}, new double[]{1}, new double[]{1}, new double[]{0}};
		double[][] trainingTargets = new double[][]{new double[]{0}, new double[]{0}, new double[]{1}, new double[]{1}};
		double[][] testData = new double[][]{new double[]{0}, new double[]{1}, new double[]{1}, new double[]{1}};
		double[][] testTargets = new double[][]{new double[]{1}, new double[]{0}};

		int j = 0;
		int counter = 0;
		double[][] inputSeq = null;
		List<SequenceM21> trainingSeq = new ArrayList<SequenceM21>();
		for(double[] train : trainingData){
			if(j==0){
				inputSeq = new double[2][];
			}
			inputSeq[j] = train;
			j++;
			if(j>=2){
				SequenceM21 seq = new SequenceM21(inputSeq, trainingTargets[counter]);
				trainingSeq.add(seq);
				j=0;
				counter++;
			}		
		}

		System.err.println("done.");
		int epoch = 0;
		double correctlyClassified;
		int batchSize = trainingData.length/100;
		do {
			epoch++;
			double trainingError = nn.sgdTrainSeq(trainingSeq, 0.04, batchSize, false, 0.4);
			int ce = ((int)(Math.exp(-trainingError)*100));
			System.out.println("epoch "+epoch+" training error: "+trainingError+" (confidence "+ce+"%)");
			correctlyClassified = test(nn, testData, testTargets);
			System.out.println((int)(correctlyClassified*100)+"% correctly classified");
		} while (correctlyClassified < 0.9);
	}

}