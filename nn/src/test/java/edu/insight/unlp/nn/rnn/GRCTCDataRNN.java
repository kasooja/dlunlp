package edu.insight.unlp.nn.rnn;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.StringTokenizer;

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.util.SerializationUtils;

import weka.core.Instance;
import weka.core.Instances;
import edu.insight.unlp.nn.NNLayer;
import edu.insight.unlp.nn.RNN;
import edu.insight.unlp.nn.af.Linear;
import edu.insight.unlp.nn.af.Sigmoid;
import edu.insight.unlp.nn.common.Sequence;
import edu.insight.unlp.nn.ef.SquareErrorFunction;
import edu.insight.unlp.nn.mlp.FullyConnectedLayer;
import edu.insight.unlp.nn.utils.BasicFileTools;

public class GRCTCDataRNN {
	private static List<Sequence> trainingData = new ArrayList<Sequence>();
	private static List<Sequence> testData = new ArrayList<Sequence>();
	//private static File gModel = new File("/Users/kartik/Work/dhundo-dobara/Corpus/ML/Corpus/GoogleNews-vectors-negative300.bin.gz");
	private static Word2Vec vec = null; 
	//SerializationUtils.readObject(new File("src/test/resources/data/Sequence/suggestion/word2vecElectronics.model"));
	private static double[] actualClassTotals = new double[9]; 
	private static double[] actualClassTrainingTotals = new double[9]; 
	private static double[] predictedCorrectClassTotals = new double[9]; 
	private static double[] predictedTotalClassTotals = new double[9];
	private static Map<String, double[]> tokenVectorMap = new HashMap<String, double[]>();

	static {
		//		try {
		//			vec = WordVectorSerializer.loadGoogleModel(gModel, true);
		//		} catch (IOException e) {
		//			e.printStackTrace();
		//		}	
	}

	public static double test(RNN network, List<Sequence> testData) {
		for(int m=0; m<predictedCorrectClassTotals.length; m++){
			predictedCorrectClassTotals[m] = 0;
			predictedTotalClassTotals[m] = 0;
		}
		int correct = 0;
		for(Sequence seq : testData){
			double[][] output = network.output(seq.inputSeq);
			double[] networkOutput = output[output.length-1];
			double[] actualOutput = seq.target[seq.target.length - 1];
			//int winnerIndex = 0;
			//	double max = Double.MIN_VALUE;
			//			for(int i=0; i<output.length; i++) {
			//				if(output[i]>max){
			//					max = output[i];
			//					winnerIndex = i; 
			//				}
			//			}
			for(int i=0; i<networkOutput.length; i++){
				//	if(i==winnerIndex){
				networkOutput[i] = Math.round(networkOutput[i]);
				//				} else {
				//					output[i] = 0.0;
				//				}
			}
			boolean equal = true;
			for(int i=0; i<networkOutput.length; i++){
				if(networkOutput[i] == 1.0){
					predictedTotalClassTotals[i]++;
					if(networkOutput[i] == actualOutput[i]){
						predictedCorrectClassTotals[i]++;						
					}
				}
				if(networkOutput[i] != actualOutput[i]){
					equal = false;
				}
			}
			if(equal){
				correct++;
			}			
		}
		//			predictedTotalClassTotals[winnerIndex]++;
		//			if(equal){
		//				correct++;
		//				predictedCorrectClassTotals[winnerIndex]++;  
		//			}
		network.resetActivationCounter(false);				
		//	}
		return ((double)correct)/testData.size();
	}

	public static void readData(String grctcDataFilePath){
		Instances instances = loadWekaData(grctcDataFilePath);
		for(Instance instance: instances){
			double[] target = new double[9];	
			String text = null;
			for(int i=0; i<instance.numAttributes()-2; i++){
				if(!instance.attribute(i).isString()) {
					double value = instance.value(instance.attribute(i));
					target[i] = value;
				} else {
					text = instance.stringValue(instance.attribute(i)).toLowerCase();
					//System.out.println(text);
				}
			}
			text = text.toLowerCase();
			StringTokenizer tokenizer = new StringTokenizer(text);
			List<double[]> inputWordVectors = new ArrayList<double[]>();
			while(tokenizer.hasMoreTokens()){
				String token = tokenizer.nextToken().trim();
				if(!token.equals("") && !token.matches(".*lrb.*") && !token.matches(".*rrb.*")) {
					double[] wordVector = null;
					if(tokenVectorMap.containsKey(token)){
						wordVector = tokenVectorMap.get(token);
						//System.out.println(wordVector.length);
					} else {
						wordVector = vec.getWordVector(token);
					}
					if(wordVector!=null){
						inputWordVectors.add(wordVector);
						tokenVectorMap.put(token, wordVector);
					}
				}
			}
			double[][] inputSeq = new double[inputWordVectors.size()][];
			inputSeq = inputWordVectors.toArray(inputSeq);
			double[][] targetSeq = new double[inputWordVectors.size()][];
			for(int k=0; k<targetSeq.length; k++) {
				targetSeq[k] = target;
			}
			targetSeq[targetSeq.length-1] = target;
			Sequence seq = new Sequence(inputSeq, targetSeq);
			int[] randArray = new Random().ints(1, 0, 16).toArray();
			if(randArray[0] == 0){
				testData.add(seq);
				for(int j=0; j<target.length; j++){
					if(target[j] == 1.0){
						actualClassTotals[j]++;
					}
				}
			} else {
				trainingData.add(seq);
				for(int j=0; j<target.length; j++){
					if(target[j] == 1.0){
						actualClassTrainingTotals[j]++;
					}
				}
			}
		}
	}

	public static Instances loadWekaData(String filePath){
		File file = new File(filePath);
		BufferedReader reader = BasicFileTools.getBufferedReader(file);
		try {
			return new Instances(reader);
		} catch (IOException e) {
			e.printStackTrace();
		}
		return null;
	}

	public static void main(String[] args) {
		RNNImpl nn = new RNNImpl(new SquareErrorFunction());
		//RNN nn = new RNN(new CrossEntropyErrorFunction());
		FullyConnectedRNNLayer outputLayer = new FullyConnectedRNNLayer(9, new Sigmoid(), nn);
		FullyConnectedRNNLayer hiddenLayer1 = new FullyConnectedRNNLayer(25, new Sigmoid(), nn);
		FullyConnectedRNNLayer hiddenLayer = new FullyConnectedRNNLayer(70, new Sigmoid(), nn);
		FullyConnectedLayer inputLayer = new FullyConnectedLayer(300, new Linear(), nn);
		List<NNLayer> layers = new ArrayList<NNLayer>();
		layers.add(inputLayer);
		layers.add(hiddenLayer);
		layers.add(hiddenLayer1);
		layers.add(outputLayer);
		nn.setLayers(layers);
		nn.initializeNN();

		System.err.print("Reading serialized word vectors...");
		tokenVectorMap = SerializationUtils.readObject(new File("src/test/resources/data/Sequence/grctc/grctcDataWordVectorMap.vecMap"));
		System.err.print("Done");

		System.err.print("Reading data...");
		String grctcDataFilePath = "src/test/resources/data/Sequence/grctc/USUKAMLAll9Labels_all.arff";
		readData(grctcDataFilePath);
		System.err.println("done.");

		//		System.err.print("Serializing word vectors...");
		//		SerializationUtils.saveObject(tokenVectorMap, new File("/Users/kartik/git/dlunlp/nn/src/main/resources/data/Sequence/grctcDataWordVectorMap.vecMap"));
		//		System.err.println("done.");

		System.out.println("TestDataSize: " + testData.size());
		System.out.println("TrainingDataSize: " + trainingData.size());
		int epoch = 0;
		double correctlyClassified;
		do {
			epoch++;
			double trainingError = nn.sgdTrain(trainingData, 0.001, true);
			//int ce = ((int)(Math.exp(-trainingError)*100));
			System.out.println("epoch "+epoch+" training error: "+trainingError);
			correctlyClassified = test(nn, testData);
			int classIndex = 0;
			for(int m=0; m<predictedCorrectClassTotals.length; m++){
				System.out.print("Class " + (classIndex+1) + ": ");
				System.out.print("Precision: " + predictedCorrectClassTotals[classIndex]/predictedTotalClassTotals[classIndex] + " ");
				System.out.println("Recall: " + predictedCorrectClassTotals[classIndex]/actualClassTotals[classIndex]);
				classIndex++;
			}			
			System.out.println("Overall Accuracy: " + (int)(correctlyClassified*100));
		} while (correctlyClassified < 1.0);
	}
}
