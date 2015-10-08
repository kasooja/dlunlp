package edu.insight.unlp.nn.data;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import au.com.bytecode.opencsv.CSVReader;
import edu.insight.unlp.nn.DataSet;
import edu.insight.unlp.nn.ErrorFunction;
import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.common.Sequence;
import edu.insight.unlp.nn.common.nlp.Word2Vector;
import edu.insight.unlp.nn.ef.SquareErrorFunction;

public class SuggestionClassificationData extends DataSet{

	private Word2Vector word2vec;
	private double[] actualClassTestTotals = new double[2]; 
	private double[] actualClassTrainingTotals = new double[2];
	private double[] predictedCorrectClassTotals = new double[2];
	private double[] predictedTotalClassTotals = new double[2];
	private Map<String, double[]> tokenVectorMap = new HashMap<String, double[]>();
	private int trainTestRatioConstant = 10;
	private String electronicsDataFilePath = "src/test/resources/data/Sequence/suggestion/electronics.csv";
	private static ErrorFunction reportingLoss = new SquareErrorFunction();

	public SuggestionClassificationData(Word2Vector word2vec) {
		this.word2vec = word2vec;
		setDataSet();
	}

	public void setDataSet(){
		training = new ArrayList<Sequence>();
		testing = new ArrayList<Sequence>();
		System.err.print("Reading data...");
		readData(electronicsDataFilePath);
		System.err.print("Done");
		setDimensions();
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
	private void readData(String dataFilePath){
		Set<String> notFound = new HashSet<String>();
		try {
			CSVReader reader = new CSVReader(new InputStreamReader(new FileInputStream(dataFilePath), "UTF-8"));
			String[] values = reader.readNext();
			while(values != null ) {
				if(values[0].equalsIgnoreCase("id")){
					values = reader.readNext();
					continue;
				}
				String label = values[3].trim();
				if(!"1".equals(label.trim())){
					label = "0";
				}
				double[] target = null;
				int classIndex = 0;
				if("0".equals(label.trim())){
					target = new double[]{1.0, 0.0};
					classIndex = 0;
				} else if("1".equals(label.trim())){
					target = new double[]{0.0, 1.0};
					classIndex = 1;
				}

				String sentence = values[1].replaceAll("\\t", "");
				if("".equals(sentence.trim())){
					continue;
				}
				sentence = sentence.toLowerCase();
				String[] tokens = sentence.replaceAll("(\\.\\.\\.+|[\\p{Po}\\p{Ps}\\p{Pe}\\p{Pi}\\p{Pf}\u2013\u2014\u2015&&[^'\\.]]|(?<!(\\.|\\.\\p{L}))\\.(?=[\\p{Z}\\p{Pf}\\p{Pe}]|\\Z)|(?<!\\p{L})'(?!\\p{L}))"," $1 ")
						.replaceAll("\\p{C}|^\\p{Z}+|\\p{Z}+$","")
						.split("\\p{Z}+");
				List<double[]> inputWordVectors = new ArrayList<double[]>();
				for(String token : tokens){
					double[] wordVector = word2vec.getWordVector(token);
					if(tokenVectorMap.containsKey(token)){
						wordVector = tokenVectorMap.get(token);
					} else {
						wordVector = word2vec.getWordVector(token);
					}
					if(wordVector!=null){
						inputWordVectors.add(wordVector);	
						tokenVectorMap.put(token, wordVector);
					} else {
						notFound.add(token);
					}
				}
				double[][] inputSeq = new double[inputWordVectors.size()][];
				inputSeq = inputWordVectors.toArray(inputSeq);
				double[][] targetSeq = new double[inputWordVectors.size()][];
				for(int k=0; k<targetSeq.length; k++) {
					targetSeq[k] = null; //make it null, if you just want to use the error at the last step
				}
				targetSeq[targetSeq.length-1] = target;
				Sequence seq = new Sequence(inputSeq, targetSeq);
				int[] randArray = new Random().ints(1, 0, trainTestRatioConstant).toArray();
				if(randArray[0] == 0){
					testing.add(seq);
					actualClassTestTotals[classIndex]++;					
				} else {
					training.add(seq);
					actualClassTrainingTotals[classIndex]++;
				}
				values = reader.readNext();
			}
			reader.close();	
		} catch (IOException e1) {
			e1.printStackTrace();
		}

		//Removing the bias
		Map<Integer, Sequence> singleClassSequence = new HashMap<Integer, Sequence>();
		for(Sequence seq : training){
			double[] target = seq.target[seq.target.length-1];
			int howMany = 0;
			int whichOne = 0;
			for(int ind=0; ind<target.length; ind++){
				if(target[ind] == 1.0){
					howMany++;
					whichOne = ind;
				}
			}
			if(howMany==1){
				singleClassSequence.put(whichOne, seq);
			}
		}
		int maxIndex = 0;
		double maxVal = Double.MIN_VALUE;
		for(int counter = 0; counter<actualClassTrainingTotals.length; counter++){
			if(actualClassTrainingTotals[counter]>maxVal) {
				maxVal = actualClassTrainingTotals[counter];
				maxIndex = counter;
			}
		}
		double maxCount = actualClassTrainingTotals[maxIndex];
		for(Integer classIndex : singleClassSequence.keySet()){
			if(actualClassTrainingTotals[classIndex]<maxCount){
				double diff = maxCount - actualClassTrainingTotals[classIndex];
				for(int val=0; val<diff; val++){
					Sequence seq = singleClassSequence.get(classIndex);
					training.add(seq);
					actualClassTrainingTotals[classIndex]++;
				}
			}
		}

		for(String token : notFound){
			if(!token.matches(".*\\d.*")){
				System.out.println(token);
			}
		}
		System.err.println("\nNot Found:   " + notFound.size() + "/" + (tokenVectorMap.size() + notFound.size()));
	}

	@Override
	public String evaluateTest(NN nn) {
		StringBuilder report = new StringBuilder();
		int totalSteps = 0;
		int totalCorrect = 0;
		for(Sequence seq : testing){
			double[][] output = nn.ff(seq, reportingLoss, false);
			double[] networkOutput = output[output.length -1];
			double[] actualOutput = seq.target[seq.target.length-1];
			int winnerIndex = 0;
			double max = Double.MIN_VALUE;
			totalSteps = totalSteps + seq.inputSeq.length;
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
				totalCorrect++;
				predictedCorrectClassTotals[winnerIndex]++;  
			}
			nn.resetActivationCounter(false);				
		}
		double correctlyClassified = ((double)totalCorrect/(double)totalSteps) * 100;  
		for(int classIndex=0; classIndex<predictedCorrectClassTotals.length; classIndex++){
			report.append("Class " + (classIndex+1) + ": ");
			report.append("Precision: " + predictedCorrectClassTotals[classIndex]/predictedTotalClassTotals[classIndex] + " ");
			report.append("Recall: " + predictedCorrectClassTotals[classIndex]/actualClassTestTotals[classIndex] + " ");
			predictedCorrectClassTotals[classIndex] = 0;
			predictedTotalClassTotals[classIndex] = 0;
		}			
		report.append("Overall Accuracy: " + (int)(correctlyClassified) + "\n");
		return report.toString();
	}

}

//private void sap(String dataFilePath){
//	try {
//		CSVReader reader = new CSVReader(new InputStreamReader(new FileInputStream(dataFilePath), "UTF-8"));
//		int count = 1;
//		String[] values = reader.readNext();
//		while(values != null ) {
//			String sentence = values[1].replaceAll("\\t", "");
//			String name = values[0].replaceAll("\\.json.*", "");
//			BasicFileTools.writeFile("src/test/resources/hotelNonSugg/" + name + count + ".txt", sentence.trim());
//			System.out.println(count++);
//			values = reader.readNext();
//		}
//		reader.close();	
//	} catch (IOException e1) {
//		e1.printStackTrace();
//	}
//}

