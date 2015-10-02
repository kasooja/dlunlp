package edu.insight.unlp.nn.data;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import edu.insight.unlp.nn.DataSet;
import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.common.Sequence;
import edu.insight.unlp.nn.utils.BasicFileTools;

public class SentimentClassificationOneHotVecCharData extends DataSet {

	private Set<String> chars = new HashSet<String>();
	private Map<String, Integer> charToIndex = new HashMap<String, Integer>();
	private double[] actualClassTestTotals = new double[2]; //binary classification, + -
	private double[] actualClassTrainingTotals = new double[2];
	private double[] predictedCorrectClassTotals = new double[2];
	private double[] predictedTotalClassTotals = new double[2];
	private String posSentDataDirPath = "src/test/resources/data/Sequence/sentiment/rt-polaritydata/rt-polarity.pos";
	private String negSentDataDirPath = "src/test/resources/data/Sequence/sentiment/rt-polaritydata/rt-polarity.neg";
	private int trainTestRatioConstant = 10;

	public SentimentClassificationOneHotVecCharData() {
		setDataSet();
	}

	@Override
	public void setDataSet(){
		double[] posTarget = new double[]{1};
		double[] negTarget = new double[]{0};
		training = new ArrayList<Sequence>();
		testing = new ArrayList<Sequence>();
		readDataCharMap(posSentDataDirPath);
		readDataCharMap(negSentDataDirPath);
		readData(posSentDataDirPath, posTarget);
		readData(negSentDataDirPath, negTarget);
		charToIndex = null;
		chars =  null;
		setDimensions();
	}
	
	private void setDimensions(){
		inputUnits = training.get(0).inputSeq[0].length;
		for(Sequence seq : training) {
			if(seq.target!=null){		
				outputUnits = seq.target[0].length;
				break;
			}
		}
	}

	private void readDataCharMap(String dataFilePath){
		BufferedReader br = BasicFileTools.getBufferedReader(dataFilePath);
		String line = null;
		int id = chars.size();
		try {
			while((line = br.readLine())!=null){
				line = line.toLowerCase();				
				for (int i = 0; i < line.length(); i++) {
					String ch = line.charAt(i) + "";
					if (chars.contains(ch) == false) {
						if (ch.equals("\n")) {
							//System.out.print("\\n");
						}
						else {
							//System.out.print(ch);
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

	private void readData(String dataFilePath, double[] target){
		BufferedReader br = BasicFileTools.getBufferedReader(dataFilePath);
		String line = null;
		try {
			int dimensions = charToIndex.size();
			br = BasicFileTools.getBufferedReader(dataFilePath);
			while((line = br.readLine())!=null){
				line = line.toLowerCase();
				List<double[]> inputCharVectors = new ArrayList<double[]>();
				for (int i = 0; i < line.length(); i++) {
					String ch = line.charAt(i) + "";
					int index = charToIndex.get(ch);
					double[] charVector = new double[dimensions];
					charVector[index] = 1.0;
					inputCharVectors.add(charVector);
				}
				int classIndex = 0;
				if(target[0] == 1.0){
					classIndex = 1;
				} 		
				double[][] targetSeq = new double[inputCharVectors.size()][];
				for(int k=0; k<targetSeq.length; k++) {
					targetSeq[k] = target;
				}
				//targetSeq[targetSeq.length-1] = target;
				double[][] inputSeq = new double[inputCharVectors.size()][];
				inputSeq = inputCharVectors.toArray(inputSeq);
				Sequence seq = new Sequence(inputSeq, targetSeq);
				int[] randArray = new Random().ints(1, 0, trainTestRatioConstant).toArray();
				if(randArray[0] == 0){
					testing.add(seq);
					actualClassTestTotals[classIndex]++;					
				} else {
					training.add(seq);
					actualClassTrainingTotals[classIndex]++;
				}
			}
		} catch (IOException e1) {
			e1.printStackTrace();
		}
	}

	@Override
	public String evaluateTest(NN nn) {
		StringBuilder report = new StringBuilder();
		int totalSteps = 0;
		int totalCorrect = 0;
		for(Sequence seq : testing){
			double[][] output = nn.output(seq.inputSeq);
			double[] actualOutput = seq.target[seq.target.length-1];
			double[] netOutput = output[output.length-1];
			totalSteps = totalSteps + seq.inputSeq.length;
			int winnerIndex = 0;
			for(int i=0; i<netOutput.length; i++) {
				netOutput[i] = Math.round(netOutput[i]); 
			}
			if(netOutput[0] == 1.0){
				winnerIndex = 1;
			} else {
				winnerIndex = 0;
			}
			boolean equal = true;
			for(int i=0; i<netOutput.length; i++){
				if(netOutput[i] != actualOutput[i]){
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
