package edu.insight.unlp.nn.rnn;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.StringTokenizer;

import org.deeplearning4j.util.SerializationUtils;

import edu.insight.unlp.nn.NNLayer;
import edu.insight.unlp.nn.RNN;
import edu.insight.unlp.nn.af.Linear;
import edu.insight.unlp.nn.af.Sigmoid;
import edu.insight.unlp.nn.common.Sequence;
import edu.insight.unlp.nn.ef.SquareErrorFunction;
import edu.insight.unlp.nn.lstm.FullyConnectedLSTMLayer;
import edu.insight.unlp.nn.mlp.FullyConnectedFFLayer;
import edu.insight.unlp.nn.utils.BasicFileTools;
//import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;

public class SentimentDataRNN {

	static Set<String> chars = new HashSet<String>();
	static Map<String, Integer> charToIndex = new HashMap<String, Integer>();

	private static List<Sequence> trainingData = new ArrayList<Sequence>();
	private static List<Sequence> testData = new ArrayList<Sequence>();
	//private static Word2Vec vec = null;// = SerializationUtils.readObject(new File("src/test/resources/data/Sequence/suggestion/word2vecElectronics.model"));
	//private static File gModel = new File("/Users/kartik/Work/dhundo-dobara/Corpus/ML/Corpus/GoogleNews-vectors-negative300.bin.gz");
	private static Map<String, double[]> vecs = null;//= HLBLVectors.vecs;
	private static double[] actualClassTestTotals = new double[2]; //binary classification, + -
	private static double[] actualClassTrainingTotals = new double[2];
	private static double[] predictedCorrectClassTotals = new double[2];
	private static double[] predictedTotalClassTotals = new double[2];
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
			double[] actualOutput = seq.target[seq.target.length-1];
			double[] netOutput = output[output.length-1];
			int winnerIndex = 0;
			//double max = Double.MIN_VALUE;
			//long actualO = Math.round(actualOutput[0]);

			for(int i=0; i<netOutput.length; i++) {
				netOutput[i] = Math.round(netOutput[i]); 
			}

			if(netOutput[0] == 1.0){
				winnerIndex = 1;
			} else {
				winnerIndex = 0;
			}

			//			for(int i=0; i<netOutput.length; i++) {
			//				if(netOutput[i]>max){
			//					max = netOutput[i];
			//					winnerIndex = i; 
			//				}
			//			}
			//			for(int i=0; i<netOutput.length; i++){
			//				if(i==winnerIndex){
			//					netOutput[i] = 1.0;
			//				} else {
			//					netOutput[i] = 0.0;
			//				}
			//			}
			boolean equal = true;
			for(int i=0; i<netOutput.length; i++){
				if(netOutput[i] != actualOutput[i]){
					equal = false;
				}
			}

			predictedTotalClassTotals[winnerIndex]++;
			if(equal){
				correct++;
				predictedCorrectClassTotals[winnerIndex]++;  
			}
			network.resetActivationCounter(false);				
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
					String token = tokenizer.nextToken().trim().toLowerCase();
					double[] wordVector = null;
					if(token.equals("")) {
						continue;
					}
					if(tokenVectorMap.containsKey(token)){
						wordVector = tokenVectorMap.get(token);
						//						System.out.print(token + " ");
						//						for(double val : wordVector){
						//							System.out.print(val + " ");
						//						}
						//System.out.println();
					} else {
						wordVector = vecs.get(token);//getWordVector(token);
					}
					if(wordVector!=null){
						inputWordVectors.add(wordVector);
						tokenVectorMap.put(token, wordVector);
					} else {
						wordVector = vecs.get("*UNKNOWN*");
						inputWordVectors.add(wordVector);
						tokenVectorMap.put(token, wordVector);
					}
				}
				int classIndex = 0; //binary classification, only 1 class would be positive
				//				for(int j=0; j<target.length; j++){
				//					if(target[j] == 1.0){
				//						classIndex = j;
				//					}
				//				}

				if(target[0] == 1.0){
					classIndex = 1;
				} 		
				double[][] targetSeq = new double[inputWordVectors.size()][];
				for(int k=0; k<targetSeq.length; k++) {
					targetSeq[k] = target;
				}
				targetSeq[targetSeq.length-1] = target;
				double[][] inputSeq = new double[inputWordVectors.size()][];
				inputSeq = inputWordVectors.toArray(inputSeq);
				Sequence seq = new Sequence(inputSeq, targetSeq);
				int[] randArray = new Random().ints(1, 0, 14).toArray();
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

	public static void convertData(String dataFilePath, double[] target){
		BufferedReader br = BasicFileTools.getBufferedReaderFile(dataFilePath);
		String line = null;
		try {
			while((line = br.readLine())!=null){
				line = line.toLowerCase();
				StringTokenizer tokenizer = new StringTokenizer(line);
				List<double[]> inputWordVectors = new ArrayList<double[]>();
				while(tokenizer.hasMoreTokens()){
					String token = tokenizer.nextToken().trim().toLowerCase();
					double[] wordVector = null;
					if(token.equals("")) {
						continue;
					}
					if(tokenVectorMap.containsKey(token)){
						wordVector = tokenVectorMap.get(token);
						//						System.out.print(token + " ");
						//						for(double val : wordVector){
						//							System.out.print(val + " ");
						//						}
						//System.out.println();
					} else {
						wordVector = vecs.get(token);//getWordVector(token);
					}
					if(wordVector!=null){
						inputWordVectors.add(wordVector);
						tokenVectorMap.put(token, wordVector);
					} else {
						wordVector = vecs.get("*UNKNOWN*");
						inputWordVectors.add(wordVector);
						tokenVectorMap.put(token, wordVector);
					}
				}
				int classIndex = 0; //binary classification, only 1 class would be positive
				for(int j=0; j<target.length; j++){
					if(target[j] == 1.0){
						classIndex = j;
					}
				}
				double[][] targetSeq = new double[inputWordVectors.size()][];
				for(int k=0; k<targetSeq.length; k++) {
					targetSeq[k] = target;
				}
				targetSeq[targetSeq.length-1] = target;
				double[][] inputSeq = new double[inputWordVectors.size()][];
				inputSeq = inputWordVectors.toArray(inputSeq);
				Sequence seq = new Sequence(inputSeq, targetSeq);
				int[] randArray = new Random().ints(1, 0, 14).toArray();
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


	public static void readDataOneHotCharMap(String dataFilePath){
		BufferedReader br = BasicFileTools.getBufferedReaderFile(dataFilePath);
		String line = null;
		int id = chars.size();
		try {
			while((line = br.readLine())!=null){
				line = line.toLowerCase();				
				for (int i = 0; i < line.length(); i++) {
					String ch = line.charAt(i) + "";
					if (chars.contains(ch) == false) {
						if (ch.equals("\n")) {
							System.out.print("\\n");
						}
						else {
							System.out.print(ch);
						}
						chars.add(ch);
						charToIndex.put(ch, id);
						id++;
					}
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void readDataOneHotChar(String dataFilePath, double[] target){
		BufferedReader br = BasicFileTools.getBufferedReaderFile(dataFilePath);
		String line = null;
		try {
			int dimensions = charToIndex.size();
			//				StringTokenizer tokenizer = new StringTokenizer(line);
			//				while(tokenizer.hasMoreTokens()){
			//					String token = tokenizer.nextToken().trim().toLowerCase();
			//					//		if(!token.equals("")){
			//					if(!tokens.contains(token)){
			//						tokens.add(token);
			//						tokenToIndex.put(token, id++);
			//					}
			//					//		}
			//				}					
			//			}
			br = BasicFileTools.getBufferedReaderFile(dataFilePath);
			while((line = br.readLine())!=null){
				line = line.toLowerCase();
				List<double[]> inputWordVectors = new ArrayList<double[]>();
				for (int i = 0; i < line.length(); i++) {
					String ch = line.charAt(i) + "";
					int index = charToIndex.get(ch);
					double[] charVector = new double[dimensions];
					charVector[index] = 1.0;
					inputWordVectors.add(charVector);
				}
				//				StringTokenizer tokenizer = new StringTokenizer(line);
				//				List<double[]> inputWordVectors = new ArrayList<double[]>();
				//				while(tokenizer.hasMoreTokens()){
				//					String token = tokenizer.nextToken().trim().toLowerCase();
				//					double[] wordVector = new double[tokenToIndex.size()];
				//					for(int m=0; m<tokenToIndex.size(); m++){
				//						if(tokenToIndex.get(token) == m){
				//							wordVector[m] = 1;
				//						}
				//					}
				//					inputWordVectors.add(wordVector);
				//				}
				int classIndex = 0; //binary classification, only 1 class would be positive
				//				for(int j=0; j<target.length; j++){
				//					if(target[j] == 1.0){
				//						classIndex = j;
				//					}
				//				}
				if(target[0] == 1.0){
					classIndex = 1;
				} 		
				double[][] targetSeq = new double[inputWordVectors.size()][];
				for(int k=0; k<targetSeq.length; k++) {
					targetSeq[k] = target;
				}
				targetSeq[targetSeq.length-1] = target;
				double[][] inputSeq = new double[inputWordVectors.size()][];
				inputSeq = inputWordVectors.toArray(inputSeq);
				Sequence seq = new Sequence(inputSeq, targetSeq);
				int[] randArray = new Random().ints(1, 0, 10).toArray();
				if(randArray[0] == -1){
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
		RNNImpl nn = new RNNImpl(new SquareErrorFunction());
		//RNN nn = new RNN(new CrossEntropyErrorFunction());
		NNLayer outputLayer = new FullyConnectedFFLayer(1, new Sigmoid(), nn);
		NNLayer hiddenLayer = new FullyConnectedLSTMLayer(35, new Sigmoid(), nn);
		NNLayer inputLayer = new FullyConnectedFFLayer(62, new Linear(), nn);
		List<NNLayer> layers = new ArrayList<NNLayer>();
		layers.add(inputLayer);
		layers.add(hiddenLayer);
		layers.add(outputLayer);
		nn.setLayers(layers);
		
		nn.initializeNN();
		System.err.print("Reading serialized word vectors...");
		tokenVectorMap = SerializationUtils.readObject(new File("src/test/resources/data/Sequence/sentiment/sentimentWikiWordHLBL.vecMap"));
		System.err.print("Done");

		System.err.print("Reading data...");
		String posSentDataDirPath = "src/test/resources/data/Sequence/sentiment/rt-polaritydata/rt-polarity.pos";
		String negSentDataDirPath = "src/test/resources/data/Sequence/sentiment/rt-polaritydata/rt-polarity.neg";
		double[] posTarget = new double[]{1};//, 1};
		double[] negTarget = new double[]{0};//, 0};
		readDataOneHotCharMap(posSentDataDirPath);
		readDataOneHotCharMap(negSentDataDirPath);
		readDataOneHotChar(posSentDataDirPath, posTarget);
		readDataOneHotChar(negSentDataDirPath, negTarget);

		//	convertData(posSentDataDirPath, posTarget);
		//	convertData(negSentDataDirPath, negTarget);

		System.out.println("TestDataSize: " + testData.size());
		System.out.println("TrainingDataSize: " + trainingData.size());
		System.err.println("done.");

		//		System.err.print("Serializing word vectors...");
		//		SerializationUtils.saveObject(tokenVectorMap, new File("src/test/resources/data/Sequence/sentiment/sentimentWikiWordHLBL.vecMap"));
		//		System.err.println("done.");
		int epoch = 0;
		double correctlyClassified = 0;
		do {
			epoch++;
			double trainingError = nn.sgdTrain(trainingData, 0.0001, false);
			//double trainingError = nn.sgdTrainSeqErrorAtLast(trainingData, 0.00001, batchSize, true, momentum);
			//int ce = ((int)(Math.exp(-trainingError)*100));
			System.out.println("epoch "+epoch+" training error: "+trainingError);
			//correctlyClassified = test(nn, testData);
			for(int classIndex=0; classIndex<predictedCorrectClassTotals.length; classIndex++){
				System.out.print("Class " + (classIndex+1) + ": ");
				System.out.print("Precision: " + predictedCorrectClassTotals[classIndex]/predictedTotalClassTotals[classIndex] + " ");
				System.out.println("Recall: " + predictedCorrectClassTotals[classIndex]/actualClassTestTotals[classIndex]);
			}			
			System.out.println("Overall Accuracy: " + (int)(correctlyClassified*100));
		} while (correctlyClassified < 1.0);
	}
}
