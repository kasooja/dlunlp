package edu.insight.unlp.nn.rnn;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.StringTokenizer;

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.util.SerializationUtils;

import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.NNLayer;
import edu.insight.unlp.nn.af.Sigmoid;
import edu.insight.unlp.nn.common.SequenceM21;
import edu.insight.unlp.nn.ef.SquareErrorFunction;
import edu.insight.unlp.nn.rnn.FullyConnectedRNNLayer;
import edu.insight.unlp.nn.rnn.InputLayerRNN;
import edu.insight.unlp.nn.rnn.RNN;
import edu.insight.unlp.nn.utils.BasicFileTools;

public class SentimentDataRNN {

	private static List<SequenceM21> trainingData = new ArrayList<SequenceM21>();
	private static List<SequenceM21> testData = new ArrayList<SequenceM21>();// = new double[][]{new double[]{0}, new double[]{1}, new double[]{1}, new double[]{1}};
	private static Word2Vec vec = SerializationUtils.readObject(new File("src/test/resources/data/Sequence/suggestion/word2vecElectronics.model"));
	//private static File gModel = new File("/Users/kartik/Work/dhundo-dobara/Corpus/ML/Corpus/GoogleNews-vectors-negative300.bin.gz");
	//private static Word2Vec vec = null; 
	private static double[] actualClassTestTotals = new double[2]; //last one to hold the overall totals
	private static double[] actualClassTrainingTotals = new double[2]; //last one to hold the overall totals
	private static double[] predictedCorrectClassTotals = new double[2]; //last one to hold the overall totals
	private static double[] predictedTotalClassTotals = new double[2]; //last one to hold the overall totals

	//	static {
	//		try {
	//			vec = WordVectorSerializer.loadGoogleModel(gModel, true);
	//		} catch (IOException e) {
	//			e.printStackTrace();
	//		}	
	//	}

	public static double test(NN network, List<SequenceM21> testData) {
		for(int m=0; m<predictedCorrectClassTotals.length; m++){
			predictedCorrectClassTotals[m] = 0;
			predictedTotalClassTotals[m] = 0;
		}
		int correct = 0;
		for(SequenceM21 seq : testData){
			double[] output = network.outputSequence(seq.inputSeq);
			double[] actualOutput = seq.target;
			int winnerIndex = 0;
			double max = Double.MIN_VALUE;
			for(int i=0; i<output.length; i++) {
				if(output[i]>max){
					max = output[i];
					winnerIndex = i; 
				}
			}
			for(int i=0; i<output.length; i++){
				if(i==winnerIndex){
					output[i] = 1.0;
				} else {
					output[i] = 0.0;
				}
			}
			boolean equal = true;
			for(int i=0; i<output.length; i++){
				if(output[i] != actualOutput[i]){
					equal = false;
				}
			}
			predictedTotalClassTotals[winnerIndex]++;
			if(equal){
				correct++;
				predictedCorrectClassTotals[winnerIndex]++;  
			}
			network.resetActivationCounter();				
		}
		return ((double)correct)/testData.size();
	}

	public static void readData(String dataFilePath, double[] target){
		BufferedReader br = BasicFileTools.getBufferedReaderFile(dataFilePath);
		String line = null;
		try {
			while((line = br.readLine())!=null){
				line = line.toLowerCase();
 				StringTokenizer tokenizer = new StringTokenizer(line);
				List<double[]> inputWordVectors = new ArrayList<double[]>();
				while(tokenizer.hasMoreTokens()){
					String token = tokenizer.nextToken();
					double[] wordVector = vec.getWordVector(token);
					if(wordVector!=null){
						inputWordVectors.add(wordVector);	
					}
				}
				int classIndex = 0; //binary classification, only 1 class would be positive
				for(int j=0; j<target.length; j++){
					if(target[j] == 1.0){
						classIndex = j;
					}
				}
				double[][] inputSeq = new double[inputWordVectors.size()][];
				inputSeq = inputWordVectors.toArray(inputSeq);
				SequenceM21 seq = new SequenceM21(inputSeq, target);
				int[] randArray = new Random().ints(1, 0, 10).toArray();
				if(randArray[0] == 0){
					testData.add(seq);
					actualClassTestTotals[classIndex]++;					
				} else {
					trainingData.add(seq);
					actualClassTrainingTotals[classIndex]++;
				}
			}
		} catch (IOException e1) {
			e1.printStackTrace();
		}
	}

	public static void main(String[] args) {
		RNN nn = new RNN(new SquareErrorFunction());
		//RNN nn = new RNN(new CrossEntropyErrorFunction());
		double momentum = 0.9;
		//SoftMaxLayer softmaxLayer = new SoftMaxLayer(2, nn);
		FullyConnectedRNNLayer preOutputLayer = new FullyConnectedRNNLayer(2, new Sigmoid(), nn);
		FullyConnectedRNNLayer hiddenLayer1 = new FullyConnectedRNNLayer(15, new Sigmoid(), nn);
		FullyConnectedRNNLayer hiddenLayer = new FullyConnectedRNNLayer(25, new Sigmoid(), nn);
		InputLayerRNN inputLayer = new InputLayerRNN(10);
		List<NNLayer> layers = new ArrayList<NNLayer>();
		layers.add(inputLayer);
		layers.add(hiddenLayer);
		layers.add(hiddenLayer1);
		layers.add(preOutputLayer);
	//	layers.add(softmaxLayer);
		nn.setLayers(layers);
		nn.initializeNN();
		System.err.print("Reading data...");
		String posSentDataDirPath = "src/test/resources/data/Sequence/sentiment/rt-polaritydata/rt-polarity.pos";
		String negSentDataDirPath = "src/test/resources/data/Sequence/sentiment/rt-polaritydata/rt-polarity.neg";
		double[] posTarget = new double[]{0, 1};
		double[] negTarget = new double[]{1, 0};
		readData(posSentDataDirPath, posTarget);
		readData(negSentDataDirPath, negTarget);
		System.out.println("TestDataSize: " + testData.size());
		System.out.println("TrainingDataSize: " + trainingData.size());
		System.err.println("done.");
		int epoch = 0;
		double correctlyClassified;
		int batchSize = trainingData.size()/100;
		do {
			epoch++;
			double trainingError = nn.sgdTrainSeq(trainingData, 0.001, batchSize, false, momentum);
			int ce = ((int)(Math.exp(-trainingError)*100));
			System.out.println("epoch "+epoch+" training error: "+trainingError+" (confidence "+ce+"%)");
			correctlyClassified = test(nn, testData);
			for(int classIndex=0; classIndex<predictedCorrectClassTotals.length; classIndex++){
				System.out.print("Class " + (classIndex+1) + ": ");
				System.out.print("Precision: " + predictedCorrectClassTotals[classIndex]/predictedTotalClassTotals[classIndex] + " ");
				System.out.println("Recall: " + predictedCorrectClassTotals[classIndex]/actualClassTestTotals[classIndex]);
			}			
			System.out.println("Overall Accuracy: " + (int)(correctlyClassified*100));
		} while (correctlyClassified < 1.0);
	}
}
