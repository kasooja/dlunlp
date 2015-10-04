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

import edu.insight.unlp.nn.DataSet;
import edu.insight.unlp.nn.ErrorFunction;
import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.common.Sequence;
import edu.insight.unlp.nn.ef.SquareErrorFunction;
import edu.insight.unlp.nn.utils.BasicFileTools;

public class SentimentClassificationWordVecData extends DataSet {

	private static Word2Vec vec = null;
	private static File gModel = new File("/Users/kartik/Work/dhundo-dobara/Corpus/ML/Corpus/GoogleNews-vectors-negative300.bin.gz");
	private double[] actualClassTestTotals = new double[2]; //binary classification, + -
	private double[] actualClassTrainingTotals = new double[2];
	private double[] predictedCorrectClassTotals = new double[2];
	private double[] predictedTotalClassTotals = new double[2];
	private String posSentDataDirPath = "src/test/resources/data/Sequence/sentiment/rt-polaritydata/rt-polarity.pos";
	private String negSentDataDirPath = "src/test/resources/data/Sequence/sentiment/rt-polaritydata/rt-polarity.neg";
	private Map<String, double[]> tokenVectorMap = new HashMap<String, double[]>();
	private int trainTestRatioConstant = 10;
	public boolean readSerializedWordVecModel = true;
	public String savedSentimentDataWord2VecModel = "src/test/resources/data/Sequence/sentiment/sentimentWikiWord.vecMap";
	private static ErrorFunction reportingLoss = new SquareErrorFunction();

	public SentimentClassificationWordVecData() {
		setDataSet();
	}

	@Override
	public void setDataSet() {
		if(!readSerializedWordVecModel){
			setWord2VecModel();
		} else {
			System.err.print("Reading serialized word vectors...");
			tokenVectorMap = SerializationUtils.readObject(new File(savedSentimentDataWord2VecModel));
			System.err.print("Done");
		}
		double[] posTarget = new double[]{1};
		double[] negTarget = new double[]{0};
		readData(posSentDataDirPath, posTarget);
		readData(negSentDataDirPath, negTarget);
		inputUnits = training.get(0).inputSeq[0].length;
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

	private void setWord2VecModel(){
		try {
			vec = WordVectorSerializer.loadGoogleModel(gModel, true);
		} catch (IOException e) {
			e.printStackTrace();
		}	
	}

	private void readData(String dataFilePath, double[] target){
		BufferedReader br = BasicFileTools.getBufferedReader(dataFilePath);
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
					} else {
						wordVector = vec.getWordVector(token);
					}
					if(wordVector!=null){
						inputWordVectors.add(wordVector);
						tokenVectorMap.put(token, wordVector);
					}				}
				int classIndex = 0; //binary classification
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
			double[][] output = nn.ff(seq, reportingLoss, false);
			double[] actualOutput = seq.target[seq.target.length-1];
			double[] netOutput = output[output.length-1];
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
