
package edu.insight.unlp.nn;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.NNLayer;
import edu.insight.unlp.nn.SequenceM21;
import edu.insight.unlp.nn.af.ReLU;
import edu.insight.unlp.nn.af.Sigmoid;
import edu.insight.unlp.nn.af.Tanh;
import edu.insight.unlp.nn.ef.CrossEntropyErrorFunction;
import edu.insight.unlp.nn.ef.SquareErrorFunction;
import edu.insight.unlp.nn.mlp.FullyConnectedLayer;
import edu.insight.unlp.nn.mlp.InputLayer;
import edu.insight.unlp.nn.mlp.MLP;
import edu.insight.unlp.nn.rnn.FullyConnectedRNNLayer;
import edu.insight.unlp.nn.rnn.RNN;

public class DigitClassifier {

	public static int argmax(double[] a) {
		double max = Double.MIN_VALUE;
		int maxi = -1;
		for (int i=0; i<a.length; i++) {
			if (a[i]>max) {
				max = a[i];
				maxi = i;
			}
		}
		if (maxi == -1) {
			throw new IllegalArgumentException();
		} else {
			return maxi;
		}
	}

	public static double test(NN network, double[][] inputs, double[][] targets) {
		int correct = 0;
		for (int i=0; i<inputs.length; i++) {
			int recognizedDigit = argmax(network.output(inputs[i]));
			int actualDigit = argmax(targets[i]);
			if (recognizedDigit == actualDigit) {
				correct++;
			}
		}
		return ((double)correct)/inputs.length;
	}

	public static double[][] parseFile(BufferedReader r, int numLines, int numColumns) throws IOException {
		double[][] lines = new double[numLines][];
		int i=0;
		while (r.ready()) {
			String line = r.readLine();
			String[] columns = line.split("\\s+");
			if (columns.length != numColumns) {
				System.err.println("Warning: invalid number of columns in "+r+" line "+(i+1)+" (actual "+columns.length+", expected "+numColumns+")");
			}
			lines[i] = new double[numColumns];
			for (int j=0; j<numColumns; j++) {
				lines[i][j] = Double.parseDouble(columns[j]);
			}
			i++;
		}
		if (i != numLines) {
			System.err.println("Warning: invalid number of lines (actual "+i+", expected "+numLines+")");
		}
		return lines;
	}

	public static void main(String[] args) {
		String traindataFile = "/Users/kartik/Work/Workspaces/Workspaces/Luna/NNs/nnlearn/nnlearn.bauerNN/src/main/resources/data/DigitClassifier/trainData/traindata";
		String traintargetsFile = "/Users/kartik/Work/Workspaces/Workspaces/Luna/NNs/nnlearn/nnlearn.bauerNN/src/main/resources/data/DigitClassifier/trainData/traintargets";
		String testdataFile = "/Users/kartik/Work/Workspaces/Workspaces/Luna/NNs/nnlearn/nnlearn.bauerNN/src/main/resources/data/DigitClassifier/testData/testdata";
		String testtargetsFile = "/Users/kartik/Work/Workspaces/Workspaces/Luna/NNs/nnlearn/nnlearn.bauerNN/src/main/resources/data/DigitClassifier/testData/testtargets";
		NN nn = new MLP(new SquareErrorFunction());
		//NN nn = new MLP(new CrossEntropyErrorFunction());
		FullyConnectedLayer outputLayer = new FullyConnectedLayer(10, new Sigmoid(), nn);		
		FullyConnectedLayer hiddenLayer = new FullyConnectedLayer(5, new ReLU(), nn);		
		InputLayer inputLayer = new InputLayer(256);
		List<NNLayer> layers = new ArrayList<NNLayer>();
		layers.add(inputLayer);
		layers.add(hiddenLayer);
		layers.add(outputLayer);
		nn.setLayers(layers);
		nn.initializeNN();
		double momentum = 0.8;
		try {
			System.err.print("Reading data...");
			double[][] trainingData = parseFile(new BufferedReader(new FileReader(traindataFile)), 3000, 256);
			double[][] trainingTargets = parseFile(new BufferedReader(new FileReader(traintargetsFile)), 3000, 10);
			double[][] testData = parseFile(new BufferedReader(new FileReader(testdataFile)), 100, 256);
			double[][] testTargets = parseFile(new BufferedReader(new FileReader(testtargetsFile)), 100, 10);
			System.err.println("done.");
			int epoch = 0;
			double correctlyClassified;
			int batchSize = trainingData.length/200;
			do {
				epoch++;
				double trainingError = nn.batchgdTrain(trainingData, trainingTargets, 0.01, batchSize, true, momentum);
				int ce = ((int)(Math.exp(-trainingError)*100));
				System.out.println("epoch "+epoch+" training error: "+trainingError+" (confidence "+ce+"%)");
				correctlyClassified = test(nn, testData, testTargets);
				System.out.println((int)(correctlyClassified*100)+"% correctly classified");
			} while (correctlyClassified < 0.9);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	
	
	
	
	
	
	
	
	
	
	public static void mainRNN(String[] args) {
		String traindataFile = "/Users/kartik/Work/Workspaces/Workspaces/Luna/NNs/nnlearn/nnlearn.bauerNN/src/main/resources/data/DigitClassifier/trainData/traindata";
		String traintargetsFile = "/Users/kartik/Work/Workspaces/Workspaces/Luna/NNs/nnlearn/nnlearn.bauerNN/src/main/resources/data/DigitClassifier/trainData/traintargets";
		String testdataFile = "/Users/kartik/Work/Workspaces/Workspaces/Luna/NNs/nnlearn/nnlearn.bauerNN/src/main/resources/data/DigitClassifier/testData/testdata";
		String testtargetsFile = "/Users/kartik/Work/Workspaces/Workspaces/Luna/NNs/nnlearn/nnlearn.bauerNN/src/main/resources/data/DigitClassifier/testData/testtargets";
		RNN nn = new RNN(new SquareErrorFunction());
		//RNN nn = new RNN(new CrossEntropyErrorFunction());
		FullyConnectedRNNLayer outputLayer = new FullyConnectedRNNLayer(10, new ReLU(), nn);
		FullyConnectedRNNLayer hiddenLayer = new FullyConnectedRNNLayer(5, new ReLU(), nn);
		InputLayer inputLayer = new InputLayer(256);
		List<NNLayer> layers = new ArrayList<NNLayer>();
		layers.add(inputLayer);
		layers.add(hiddenLayer);
		layers.add(outputLayer);
		nn.setLayers(layers);
		nn.initializeNN();
		try {
			System.err.print("Reading data...");
			double[][] trainingData = parseFile(new BufferedReader(new FileReader(traindataFile)), 3000, 256);
			double[][] trainingTargets = parseFile(new BufferedReader(new FileReader(traintargetsFile)), 3000, 10);
			double[][] testData = parseFile(new BufferedReader(new FileReader(testdataFile)), 100, 256);
			double[][] testTargets = parseFile(new BufferedReader(new FileReader(testtargetsFile)), 100, 10);
			int j = 0;
			int number = 4;
			int counter = 0;
			double[][] inputSeq = null;
			List<SequenceM21> trainingSeq = new ArrayList<SequenceM21>();
			for(double[] train : trainingData){
				if(j==0){
					inputSeq = new double[number][];
				}
				if(j>=number){
					SequenceM21 seq = new SequenceM21(inputSeq, trainingTargets[counter]);
					trainingSeq.add(seq);
					int[] randSeqLengths = new Random().ints(1, 1, 5).toArray();
					number = randSeqLengths[0];
					j=0;
				}
				inputSeq[j] = train;
				counter++;	
				j++;
			}
			System.err.println("done.");
			int epoch = 0;
			double correctlyClassified;
			int batchSize = trainingData.length/200;
			do {
				epoch++;
				//double trainingError = nn.sgdTrain(trainingData, trainingTargets, 0.01);
				double trainingError = nn.sgdTrainSeq(trainingSeq, 0.01, batchSize, false, 0.8);
				int ce = ((int)(Math.exp(-trainingError)*100));
				System.out.println("epoch "+epoch+" training error: "+trainingError+" (confidence "+ce+"%)");
				correctlyClassified = test(nn, testData, testTargets);
				System.out.println((int)(correctlyClassified*100)+"% correctly classified");
			} while (correctlyClassified < 0.9);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}


}