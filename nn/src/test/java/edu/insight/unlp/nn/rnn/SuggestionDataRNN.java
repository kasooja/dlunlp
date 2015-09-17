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
import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.NNLayer;
import edu.insight.unlp.nn.af.ReLU;
import edu.insight.unlp.nn.af.Sigmoid;
import edu.insight.unlp.nn.common.SequenceM21;
import edu.insight.unlp.nn.common.SoftMaxLayer;
import edu.insight.unlp.nn.ef.SquareErrorFunction;
import edu.insight.unlp.nn.rnn.FullyConnectedRNNLayer;
import edu.insight.unlp.nn.rnn.InputLayerRNN;
import edu.insight.unlp.nn.rnn.RNN;

public class SuggestionDataRNN {

	private static List<SequenceM21> trainingData = new ArrayList<SequenceM21>();
	private static List<SequenceM21> testData = new ArrayList<SequenceM21>();// = new double[][]{new double[]{0}, new double[]{1}, new double[]{1}, new double[]{1}};
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
				SequenceM21 seq = new SequenceM21(inputSeq, target);
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
		RNN nn = new RNN(new SquareErrorFunction());
		//RNN nn = new RNN(new CrossEntropyErrorFunction());
		double momentum = 0.9;
		SoftMaxLayer softmaxLayer = new SoftMaxLayer(3, nn);
		FullyConnectedRNNLayer preOutputLayer = new FullyConnectedRNNLayer(3, new Sigmoid(), nn);
		FullyConnectedRNNLayer hiddenLayer1 = new FullyConnectedRNNLayer(15, new ReLU(), nn);
		FullyConnectedRNNLayer hiddenLayer = new FullyConnectedRNNLayer(25, new ReLU(), nn);
		InputLayerRNN inputLayer = new InputLayerRNN(10);
		List<NNLayer> layers = new ArrayList<NNLayer>();
		layers.add(inputLayer);
		layers.add(hiddenLayer);
		layers.add(hiddenLayer1);
		layers.add(preOutputLayer);
		layers.add(softmaxLayer);
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
				System.out.println("Recall: " + predictedCorrectClassTotals[classIndex]/actualClassTotals[classIndex]);
			}			
			System.out.println("Overall Accuracy: " + (int)(correctlyClassified*100));
		} while (correctlyClassified < 1.0);
	}
}
