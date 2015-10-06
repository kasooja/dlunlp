package edu.insight.unlp.nn.data;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.StringTokenizer;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.util.SerializationUtils;

import weka.core.Instance;
import weka.core.Instances;
import edu.insight.unlp.nn.DataSet;
import edu.insight.unlp.nn.ErrorFunction;
import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.common.Sequence;
import edu.insight.unlp.nn.ef.SquareErrorFunction;
import edu.insight.unlp.nn.utils.BasicFileTools;

public class GRCTCProvisionClassificationData extends DataSet {

	private static Word2Vec vec = null;
	private static File gModel = new File("/Users/kartik/Work/dhundo-dobara/Corpus/ML/Corpus/GoogleNews-vectors-negative300.bin.gz");
	private double[] actualClassTestTotals = new double[9]; 
	private double[] actualClassTrainingTotals = new double[9];
	private double[] predictedCorrectClassTotals = new double[9];
	private double[] predictedTotalClassTotals = new double[9];
	private int trainTestRatioConstant = 10;
	private Map<String, double[]> tokenVectorMap = new HashMap<String, double[]>();
	private boolean readSerializedWordVecGoogleModel = false; //private boolean writeGrctcDataWord2VecMap = "";
	private String savedGRCTCDataWord2VecMap = "src/test/resources/data/Sequence/grctc/grctcDataWordVectorMap.vecMap";
	private String grctcDataFilePath = "src/test/resources/data/Sequence/grctc/USUKAMLAll9Labels_all.arff";
	private static ErrorFunction reportingLoss = new SquareErrorFunction();

	public GRCTCProvisionClassificationData() {
		setDataSet();
	}

	public void setDataSet(){
		training = new ArrayList<Sequence>();
		testing = new ArrayList<Sequence>();
		if(readSerializedWordVecGoogleModel){
			setWord2VecGoogleModel();
		} else {
			System.err.print("Reading serialized GRCTC word2vec map . . .");
			setTokenVectorMap();
			System.err.println("Done");
		}
		System.err.print("Reading data...");
		readData(grctcDataFilePath);
		System.err.println("Done");
		setDimensions();
	}

	private void setWord2VecGoogleModel(){
		try {
			vec = WordVectorSerializer.loadGoogleModel(gModel, true);
		} catch (IOException e) {
			e.printStackTrace();
		}	
	}

	private void setTokenVectorMap(){
		tokenVectorMap = SerializationUtils.readObject(new File(savedGRCTCDataWord2VecMap));
	}

	private void setDimensions(){
		inputUnits = training.get(0).inputSeq[0].length;
		for(Sequence seq : training) {
			for(double[] target : seq.target){
				if(target!=null){
					outputUnits = target.length;
					break;
				}
			}
		}
	}

	private void readData(String grctcDataFilePath){
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
				targetSeq[k] = null;//target; //make it null to use the error just at last
			}
			targetSeq[targetSeq.length-1] = target;
			Sequence seq = new Sequence(inputSeq, targetSeq);
			int[] randArray = new Random().ints(1, 0, trainTestRatioConstant).toArray();
			if(randArray[0] == 0){
				testing.add(seq);
				for(int j=0; j<target.length; j++){
					if(target[j] == 1.0){
						actualClassTestTotals[j]++;
					}
				}
			} else {
				training.add(seq);
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

	@Override
	public String evaluateTest(NN nn) {
		nn.resetError();
		StringBuilder report = new StringBuilder();
		int totalSteps = 0;
		int totalCorrect = 0;
		for(Sequence seq : testing){
			double[] actualOutput = seq.target[seq.target.length - 1];
			double[][] output = nn.ff(seq, reportingLoss, false);
			double[] networkOutput = new double[actualOutput.length];
			for(int m=0; m<output.length; m++){
				for(int k=0; k<output[0].length; k++){
					networkOutput[k] = networkOutput[k] + output[m][k]; 
				}
			}
			for(int k=0; k<output[0].length; k++){
				networkOutput[k] = networkOutput[k]/output.length;
			}

			for(int i=0; i<networkOutput.length; i++){
				networkOutput[i] = Math.round(networkOutput[i]);
			}
			boolean equal = true; 
			totalSteps++;// + seq.inputSeq.length;
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
				totalCorrect++;
			}			
		}
		double correctlyClassified = ((double)totalCorrect/(double)totalSteps) * 100;  
		for(int classIndex=0; classIndex<predictedCorrectClassTotals.length; classIndex++){
			report.append("Class " + (classIndex+1) + ": ");
			report.append("Precision: " + predictedCorrectClassTotals[classIndex]/predictedTotalClassTotals[classIndex] + " ");
			report.append("Recall: " + predictedCorrectClassTotals[classIndex]/actualClassTestTotals[classIndex] + " \n");
			predictedCorrectClassTotals[classIndex] = 0;
			predictedTotalClassTotals[classIndex] = 0;
		}
		report.append("Test Loss: " + nn.getError() + "\n");
		report.append("Overall Accuracy: " + (int)(correctlyClassified) + "\n");
		nn.resetError();
		return report.toString();
	}

}