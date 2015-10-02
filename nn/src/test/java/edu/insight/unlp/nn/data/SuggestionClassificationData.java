package edu.insight.unlp.nn.data;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.StringTokenizer;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.util.SerializationUtils;

import au.com.bytecode.opencsv.CSVReader;
import edu.insight.unlp.nn.DataSet;
import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.common.Sequence;

public class SuggestionClassificationData extends DataSet{

	private static Word2Vec vec = null;
	private static File gModel = new File("/Users/kartik/Work/dhundo-dobara/Corpus/ML/Corpus/GoogleNews-vectors-negative300.bin.gz");
	private double[] actualClassTestTotals = new double[3]; 
	private double[] actualClassTrainingTotals = new double[3];
	private double[] predictedCorrectClassTotals = new double[3];
	private double[] predictedTotalClassTotals = new double[3];
	private int trainTestRatioConstant = 10;
	private boolean readSerializedWordVecGoogleModel = false;
	private String savedSuggestionDataWord2VecModel = "src/test/resources/data/Sequence/suggestion/word2vecElectronics.model";
	private String electronicsDataFilePath = "src/test/resources/data/Sequence/suggestion/electronics.csv";
	
	public SuggestionClassificationData() {
		setDataSet();
	}

	public void setDataSet(){
		if(!readSerializedWordVecGoogleModel){
			setWord2VecGoogleModel();
		} else {
			System.err.print("Reading serialized word vectors...");
			setWord2VecSuggestionDataModel();
			System.err.print("Done");
		}
		System.err.print("Reading data...");
		readData(electronicsDataFilePath);
		System.err.print("Done");
		setDimensions();
	}

	private void setWord2VecGoogleModel(){
		try {
			vec = WordVectorSerializer.loadGoogleModel(gModel, true);
		} catch (IOException e) {
			e.printStackTrace();
		}	
	}
	
	private void setWord2VecSuggestionDataModel(){
		vec = SerializationUtils.readObject(new File(savedSuggestionDataWord2VecModel));	
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

	private void readData(String dataFilePath){
		try {
			CSVReader reader = new CSVReader(new InputStreamReader(new FileInputStream(dataFilePath), "UTF-8"));
			String[] values = reader.readNext();
			while(values != null ) {
				String sentence = values[1].replaceAll("\\t", "");
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
				if("1".equals(label.trim())){
					target = new double[]{1.0, 0.0, 0.0};
					classIndex = 0;
				}				
				if("2".equals(label.trim())){
					target = new double[]{0.0, 1.0, 0.0};
					classIndex = 1;
				}
				if("3".equals(label.trim())){
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
				//targetSeq[targetSeq.length-1] = target;
				Sequence seq = new Sequence(inputSeq, targetSeq);
				int[] randArray = new Random().ints(1, 0, trainTestRatioConstant).toArray();
				if(randArray[0] == 0){
					testing.add(seq);
					actualClassTestTotals[classIndex]++;					
				} else {
					//	if(classIndex == 1){ //to reduce the bias of a class, use this
					//	if(randArray[0]==0)	{	
					training.add(seq);
					actualClassTrainingTotals[classIndex]++;
					//		}					
					//	} else {
					//	 training.add(seq);
					//	 actualClassTrainingTotals[classIndex]++;
					//	}
				}
				values = reader.readNext();
			}
			reader.close();	
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
