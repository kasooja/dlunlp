package edu.insight.unlp.nn.rnn;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.StringTokenizer;

//import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.util.SerializationUtils;

import au.com.bytecode.opencsv.CSVReader;
import edu.insight.unlp.nn.NNLayer;
import edu.insight.unlp.nn.RNN;
import edu.insight.unlp.nn.af.Linear;
import edu.insight.unlp.nn.af.Sigmoid;
import edu.insight.unlp.nn.common.Sequence;
import edu.insight.unlp.nn.ef.SquareErrorFunction;
import edu.insight.unlp.nn.mlp.FullyConnectedLayer;

public class SuggestionDataRNN {

	private static List<Sequence> trainingData = new ArrayList<Sequence>();
	private static List<Sequence> testData = new ArrayList<Sequence>();// = new double[][]{new double[]{0}, new double[]{1}, new double[]{1}, new double[]{1}};
	private static Word2Vec vec = SerializationUtils.readObject(new File("src/test/resources/data/Sequence/suggestion/word2vecElectronics.model"));
	//private static File gModel = new File("/Users/kartik/Work/dhundo-dobara/Corpus/ML/Corpus/GoogleNews-vectors-negative300.bin.gz");
	//private static Word2Vec vec = null; 
	private static double[] actualClassTotals = new double[3]; //last one to hold the overall totals
	private static double[] actualClassTrainingTotals = new double[3]; //last one to hold the overall totals
	private static double[] predictedCorrectClassTotals = new double[3]; //last one to hold the overall totals
	private static double[] predictedTotalClassTotals = new double[3]; //last one to hold the overall totals

	//	static {
	//		try {
	//			vec = WordVectorSerializer.loadGoogleModel(gModel, true);
	//		} catch (IOException e) {
	//			e.printStackTrace();
	//		}	
	//	}

	public static double test(RNN network, List<Sequence> testData) {
		for(int m=0; m<predictedCorrectClassTotals.length; m++){
			predictedCorrectClassTotals[m] = 0;
			predictedTotalClassTotals[m] = 0;
		}
		int correct = 0;
		for(Sequence seq : testData){
			double[][] output = network.output(seq.inputSeq);
			double[] networkOutput = output[output.length -1];
			double[] actualOutput = seq.target[seq.target.length-1];
			int winnerIndex = 0;
			double max = Double.MIN_VALUE;
			for(int i=0; i<networkOutput.length; i++) {
				if(networkOutput[i]>max){
					max = networkOutput[i];
					winnerIndex = i; 
				}
			}
			for(int i=0; i<networkOutput.length; i++){
				if(i==winnerIndex){
					networkOutput[i] = 1.0;
				} else {
					networkOutput[i] = 0.0;
				}
			}
			boolean equal = true;
			for(int i=0; i<networkOutput.length; i++){
				if(networkOutput[i] != actualOutput[i]){
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

	public static void readData(String dataFilePath){
		try {
			CSVReader reader = new CSVReader(new InputStreamReader(new FileInputStream(dataFilePath), "UTF-8"));
			String[] values = reader.readNext();
			while(values != null ) {
				String sentence = values[1];
				if(values[0].equalsIgnoreCase("id")){
					values = reader.readNext();
					continue;
				}
				String label = values[2];
				double[] target = null;
				int classIndex = 0;
				//if("1".equals(label.trim())){
				if(Integer.parseInt(label.trim())>0){
					target = new double[]{1.0, 0.0, 0.0};
					classIndex = 0;
				}				
				//if("2".equals(label.trim())){
				if(Integer.parseInt(label.trim())==0){
					target = new double[]{0.0, 1.0, 0.0};
					classIndex = 1;
				}
				//if("3".equals(label.trim())){
				if(Integer.parseInt(label.trim())<0){
					target = new double[]{0.0, 0.0, 1.0};
					classIndex = 2;
				}
				sentence = sentence.toLowerCase();
				StringTokenizer tokenizer = new StringTokenizer(sentence);
				List<double[]> inputWordVectors = new ArrayList<double[]>();
				while(tokenizer.hasMoreTokens()){
					String token = tokenizer.nextToken();
					double[] wordVector = vec.getWordVector(token);
					if(wordVector!=null){
						inputWordVectors.add(wordVector);	
					}
				}
				double[][] inputSeq = new double[inputWordVectors.size()][];
				inputSeq = inputWordVectors.toArray(inputSeq);
				double[][] targetSeq = new double[inputWordVectors.size()][];
				for(int k=0; k<targetSeq.length; k++) {
					targetSeq[k] = target; //make it null, if you just want to use the error at the last step
				}
				targetSeq[targetSeq.length-1] = target;
				Sequence seq = new Sequence(inputSeq, targetSeq);
				int[] randArray = new Random().ints(1, 0, 16).toArray();
				if(randArray[0] == 0){
					testData.add(seq);
					actualClassTotals[classIndex]++;					
				} else {
					randArray = new Random().ints(1, 0, 2).toArray();
					if(classIndex == 1){
						if(randArray[0]==0)	{	
							trainingData.add(seq);
							actualClassTrainingTotals[classIndex]++;
						}					
					} else {
						trainingData.add(seq);
						actualClassTrainingTotals[classIndex]++;
					}
				}
				values = reader.readNext();
			}
			reader.close();	
		} catch (IOException e1) {
			e1.printStackTrace();
		}
	}

	public static void main(String[] args) {
		RNNImpl nn = new RNNImpl(new SquareErrorFunction());
		//RNN nn = new RNN(new CrossEntropyErrorFunction());
		FullyConnectedLayer outputLayer = new FullyConnectedLayer(3, new Sigmoid(), nn);
		FullyConnectedRNNLayer hiddenLayer1 = new FullyConnectedRNNLayer(15, new Sigmoid(), nn);
		FullyConnectedRNNLayer hiddenLayer = new FullyConnectedRNNLayer(25, new Sigmoid(), nn);
		FullyConnectedLayer inputLayer = new FullyConnectedLayer(10, new Linear(), nn);
		List<NNLayer> layers = new ArrayList<NNLayer>();
		layers.add(inputLayer);
		layers.add(hiddenLayer);
		layers.add(hiddenLayer1);
		layers.add(outputLayer);
		nn.setLayers(layers);
		nn.initializeNN();
		System.err.print("Reading data...");
		String electronicsDataFilePath = "src/test/resources/data/Sequence/suggestion/electronics.csv";
		readData(electronicsDataFilePath);
		System.out.println("TestDataSize: " + testData.size());
		System.out.println("TrainingDataSize: " + trainingData.size());
		System.err.println("done.");
		int epoch = 0;
		double correctlyClassified;
		do {
			epoch++;
			double trainingError = nn.sgdTrain(trainingData, 0.001, true);
			//int ce = ((int)(Math.exp(-trainingError)*100));
			System.out.println("epoch "+epoch+" training error: "+trainingError);
			correctlyClassified = test(nn, testData);
			for(int classIndex=0; classIndex<predictedCorrectClassTotals.length; classIndex++){
				System.out.print("Class " + (classIndex+1) + ": ");
				System.out.print("Precision: " + predictedCorrectClassTotals[classIndex]/predictedTotalClassTotals[classIndex] + " ");
				System.out.println("Recall: " + predictedCorrectClassTotals[classIndex]/actualClassTotals[classIndex]);
			}			
			System.out.println("Overall Accuracy: " + (int)(correctlyClassified*100));
		} while (correctlyClassified < 1.0);
	}
}
