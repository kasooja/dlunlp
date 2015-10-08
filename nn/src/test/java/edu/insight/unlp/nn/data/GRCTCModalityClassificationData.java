package edu.insight.unlp.nn.data;

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

import org.deeplearning4j.util.SerializationUtils;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.StringToWordVector;
import edu.insight.unlp.nn.DataSet;
import edu.insight.unlp.nn.ErrorFunction;
import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.common.Sequence;
import edu.insight.unlp.nn.common.nlp.Word2Vector;
import edu.insight.unlp.nn.ef.SquareErrorFunction;
import edu.insight.unlp.nn.utils.BasicFileTools;

public class GRCTCModalityClassificationData extends DataSet {

	private double[] actualClassTestTotals = new double[3]; 
	private double[] actualClassTrainingTotals = new double[3];
	private double[] predictedCorrectClassTotals = new double[3];
	private double[] predictedTotalClassTotals = new double[3];
	private int trainTestRatioConstant = 10;
	private Map<String, double[]> tokenVectorMap = new HashMap<String, double[]>();
	private String grctcDataFilePath = "src/test/resources/data/Sequence/grctc/USUKModalityAll.arff";
	private String grctcFeaturesDataFilePath = "src/test/resources/data/Sequence/grctc/USUKModalityAll.arff";
	private static ErrorFunction reportingLoss = new SquareErrorFunction();
	private static Word2Vector word2Vec;
	private boolean readSerializedTokenVectorMap = false;
	private static List<String> labels = new ArrayList<String>();
	private static Map<String, String> wrongWords = new HashMap<String, String>();

	static {
		wrongWords.put("personmaymake", "person may make");
		wrongWords.put("threemonths", "three months");
		wrongWords.put("certiﬁcates", "certificates");
		wrongWords.put("hemust", "he must");
		wrongWords.put("justiﬁed", "justified");
		wrongWords.put("theauthority", "the authority");
		wrongWords.put("connectionwith", "connection with");
		wrongWords.put("inwhole", "in whole");
		wrongWords.put("fromhim", "from him");
		wrongWords.put("personmay", "person may");
		wrongWords.put("bemade", "be made");
		wrongWords.put("cash.accrued", "cash . accrued");
		wrongWords.put("otherwisemade", "otherwise made");
		wrongWords.put("somuch", "so much");
		wrongWords.put("inoslvency", "insolvency");
		wrongWords.put("conﬁrm", "confirm");
		wrongWords.put("fromwhom", "from whom");
		wrongWords.put("conﬁrmed", "confirmed");
		wrongWords.put("beginningwith", "beginning with");
		wrongWords.put("sheriﬀmay", "sheriff may");
		wrongWords.put("theinterpretation", "the interpretation");
		wrongWords.put("amagistrates", "a magistrates");
		wrongWords.put("otheruse", "other use");
		wrongWords.put("laundering.concerned", "laundering . concerned");
		wrongWords.put("notiﬁcation", "notification");
		wrongWords.put("unincorporate", "un incorporate");
		wrongWords.put("ﬁnancial", "financial");
		wrongWords.put("satisﬁed", "satisfied");
		wrongWords.put("innorthern", "in northern");
		wrongWords.put("andforfeiture", "and for feiture");
		wrongWords.put("procedureswhich", "procedures which");
		wrongWords.put("terroristpermission", "terrorist permission");
		wrongWords.put("eﬀect", "effect");
		wrongWords.put("identiﬁed", "identified");
		wrongWords.put("deﬁnition", "definition");
		wrongWords.put("accountholder", "account holder");
		wrongWords.put("identiﬁes", "identifies");
		wrongWords.put("whichseizure", "which seizure");
		wrongWords.put("underdetained", "under detained");
		wrongWords.put("oﬀence", "offence");
		wrongWords.put("therepresentation", "the representation");
		wrongWords.put("courtmay", "court may");
		wrongWords.put("ﬁt", "fit");
		wrongWords.put("commmencing", "commencing");
		wrongWords.put("anymoney", "any money");
		wrongWords.put("ﬁscal", "fiscal");
		wrongWords.put("aﬀected", "affected");
	}

	//private boolean serializeTokenVectorMap = false;
	private String serializedGRCTCProvisionWord2VecGoogleMap = "src/test/resources/data/Sequence/grctc/serializedGRCTCProvisionWord2VecGoogleMap";

	public GRCTCModalityClassificationData(edu.insight.unlp.nn.common.nlp.Word2Vector word2vec) {
		labels.add("Obligation");
		labels.add("Prohibition");
		labels.add("Other");
		this.word2Vec = word2vec;
		setDataSet();
	}

	public void setDataSet(){
		training = new ArrayList<Sequence>();
		testing = new ArrayList<Sequence>();
		if(readSerializedTokenVectorMap){
			System.err.print("Reading serialized GRCTC TokenVectorMap  . . .");
			setTokenVectorMap(serializedGRCTCProvisionWord2VecGoogleMap);
			System.err.println("Done");
		}
		System.err.print("Reading data...");
		readDataExtendedFeatures(grctcFeaturesDataFilePath);
		System.err.println("Done");
		setDimensions();
	}


	private void setTokenVectorMap(String serializedGRCTCProvisionTokenVectorMap){
		tokenVectorMap = SerializationUtils.readObject(new File(serializedGRCTCProvisionTokenVectorMap));
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
		List<Sequence> tempTesting = new ArrayList<Sequence>();
		Set<String> notFound = new HashSet<String>();
		Instances instances = loadWekaData(grctcDataFilePath);
		for(Instance instance: instances){
			double[] target = new double[3];
			int value = (int) instance.value(instance.attribute(0));
			String label = labels.get(value);
			if("Obligation".equalsIgnoreCase(label)){
				target[0] = 1; target[1] = 0; target[2] = 0;
			}
			if("Prohibition".equalsIgnoreCase(label)){
				target[0] = 0; target[1] = 1; target[2] = 0;
			}
			if("Other".equalsIgnoreCase(label)){
				target[0] = 0; target[1] = 0; target[2] = 1;
			}

			String text = null;
			text = instance.stringValue(instance.attribute(1)).toLowerCase();
			text = text.toLowerCase();
			text = text.replace("-", " ").trim();
			text = text.replace("—", " ").trim();
			text = text.replace("classwekaattribute", "class").trim();
			if("".equals(text.trim())){
				continue;
			}
			for(String wrong : wrongWords.keySet()){
				text = (text.replaceAll(wrong, wrongWords.get(wrong).trim())).trim();
			}

			//StringTokenizer tokenizer = new StringTokenizer(text);
			String[] tokens = text.replaceAll("(\\.\\.\\.+|[\\p{Po}\\p{Ps}\\p{Pe}\\p{Pi}\\p{Pf}\u2013\u2014\u2015&&[^'\\.]]|(?<!(\\.|\\.\\p{L}))\\.(?=[\\p{Z}\\p{Pf}\\p{Pe}]|\\Z)|(?<!\\p{L})'(?!\\p{L}))"," $1 ")
					.replaceAll("\\p{C}|^\\p{Z}+|\\p{Z}+$","")
					.split("\\p{Z}+");
			List<double[]> inputWordVectors = new ArrayList<double[]>();
			//while(tokenizer.hasMoreTokens()){
			for(String token : tokens){
				//String token = tokenizer.nextToken().trim();
				if(!token.equals("") && !token.matches(".*lrb.*") && !token.matches(".*rrb.*")) {
					double[] wordVector = null;
					if(tokenVectorMap.containsKey(token)){
						wordVector = tokenVectorMap.get(token);
					} else {
						wordVector = word2Vec.getWordVector(token);
//						if(readSerializedWordVecGoogleModel){
//							wordVector = word2vecGoogle.getWordVector(token);
//						} else if(readGloveVectors){
//							wordVector = GloveVectors.getWordVector(token);
//						} else if(readHLBLVectors){
//							wordVector = HLBLVectors.getWordVector(token);
//						}
					}
					if(wordVector!=null){
						inputWordVectors.add(wordVector);
						tokenVectorMap.put(token, wordVector);
					} else {
						notFound.add(token);
					}
				}
			}

			double[][] inputSeq = new double[inputWordVectors.size()][];
			inputSeq = inputWordVectors.toArray(inputSeq);
			double[][] targetSeq = new double[inputWordVectors.size()][];
			for(int k=0; k<targetSeq.length; k++) {
				targetSeq[k] = null; //target; //make it null to use the error just at last
			}
			targetSeq[targetSeq.length-1] = target;
			Sequence seq = new Sequence(inputSeq, targetSeq);
			seq.labelString = label;
			int[] randArray = new Random().ints(1, 0, trainTestRatioConstant).toArray();
			if(randArray[0] == 0){
				tempTesting.add(seq);
			} else {
				int count = 0;
				if("Obligation".equalsIgnoreCase(label)){
					count = 2;
				}
				if("Prohibition".equalsIgnoreCase(label)){
					count = 10;
				}
				if("Other".equalsIgnoreCase(label)){
					count = 1;
				}
				for(int j=0; j<count; j++){
					training.add(seq);
				}
				for(int j=0; j<target.length; j++){
					if(target[j] == 1.0){
						actualClassTrainingTotals[j] = actualClassTrainingTotals[j] + count;
					}
				}
			}
		}
		//		for(String token : notFound){
		//			if(!token.matches(".*\\d.*")){
		//				System.out.println(token);
		//			}
		//		}

		int minCount = 0;
		for(Sequence seq : tempTesting){
			if(seq.labelString.equalsIgnoreCase("Prohibition")){
				minCount++;
			}
		}

		int oblAdded = 0;
		int othersAdded = 0;
		for(Sequence seq : tempTesting){
			if(seq.labelString.equalsIgnoreCase("Obligation")){
				oblAdded++;
				if(oblAdded<=minCount){
					testing.add(seq);
					double[] target = seq.target[seq.target.length-1];
					for(int j=0; j<target.length; j++){
						if(target[j] == 1.0){
							actualClassTestTotals[j]++;
						}
					}
				}
			} else if(seq.labelString.equalsIgnoreCase("Other")){
				othersAdded++;
				if(othersAdded<=minCount){
					testing.add(seq);
					double[] target = seq.target[seq.target.length-1];
					for(int j=0; j<target.length; j++){
						if(target[j] == 1.0){
							actualClassTestTotals[j]++;
						}
					}
				}
			} else {
				testing.add(seq);
				double[] target = seq.target[seq.target.length-1];
				for(int j=0; j<target.length; j++){
					if(target[j] == 1.0){
						actualClassTestTotals[j]++;
					}
				}
			}
		}
		System.err.println("\nNot Found" + notFound.size() + "/" + (tokenVectorMap.size() + notFound.size()));
	}

	private void readDataExtendedFeatures(String grctcDataFilePath){
		Set<String> notFound = new HashSet<String>();
		Instances instances = loadWekaData(grctcDataFilePath);
		Instances copyInstances = loadWekaData(grctcDataFilePath);

		AttributeSelection attrSel = Commons.getAttributeSelectionFilter();		
		StringToWordVector textStringToWordVectorFilter = Commons.getStringToWordVectorFilter();		
		StringToWordVector posStringToWordVectorFilter = Commons.getStringToWordVectorFilter();
		StringToWordVector frameStringToWordVectorFilter = Commons.getStringToWordVectorFilter();

		posStringToWordVectorFilter.setWordsToKeep(400);
		frameStringToWordVectorFilter.setWordsToKeep(400);
		textStringToWordVectorFilter.setAttributeIndices("2");

		NGramTokenizer tok = new NGramTokenizer();
		tok.setNGramMinSize(2);
		tok.setNGramMaxSize(3);		
		posStringToWordVectorFilter.setTokenizer(tok);
		posStringToWordVectorFilter.setAttributeNamePrefix("POSTAG_");		
		posStringToWordVectorFilter.setAttributeIndices("4");

		tok = new NGramTokenizer();
		tok.setNGramMinSize(1);
		tok.setNGramMaxSize(1);
		frameStringToWordVectorFilter.setAttributeNamePrefix("Frame_");		
		frameStringToWordVectorFilter.setAttributeIndices("3");		
		frameStringToWordVectorFilter.setTokenizer(tok);

		Filter remove1 = Commons.getRemoveFilterByRegex(".*lrb.*");
		Filter remove2 = Commons.getRemoveFilterByRegex(".*rrb.*");
		Filter remove3 = Commons.getRemoveFilterByRegex(".*\\d+.*");

		MultiFilter multiFilter = new MultiFilter();
		multiFilter.setFilters(new Filter[]{posStringToWordVectorFilter, frameStringToWordVectorFilter, 
				textStringToWordVectorFilter, remove1, remove2, remove3});
		Instances selectedAttsInstances = null;
		try {
			multiFilter.setInputFormat(copyInstances);
		//	copyInstances.setClass(copyInstances.attribute(0));;
			Instances filteredCopyInstances = Filter.useFilter(copyInstances,  multiFilter);
			filteredCopyInstances.setClassIndex(0);
			attrSel.setInputFormat(filteredCopyInstances);
			selectedAttsInstances = Filter.useFilter(filteredCopyInstances, attrSel);
			//System.out.println(selectedAttsInstances.toString());
		} catch (Exception e) {
			e.printStackTrace();
		}			
		
		int selectCounter = 0;
		for(Instance instance: instances){
			Instance selectedAttsInstance = selectedAttsInstances.get(selectCounter++);
			double[] target = new double[3];
			int value = (int) instance.value(instance.attribute(0));
			String label = labels.get(value);
			if("Obligation".equalsIgnoreCase(label)){
				target[0] = 1; target[1] = 0; target[2] = 0;
			}
			if("Prohibition".equalsIgnoreCase(label)){
				target[0] = 0; target[1] = 1; target[2] = 0;
			}
			if("Other".equalsIgnoreCase(label)){
				target[0] = 0; target[1] = 0; target[2] = 1;
			}

			int numAttributes = selectedAttsInstance.numAttributes();
			double[] featureExtension = new double[numAttributes-1];	
			for(int i=0; i<selectedAttsInstance.numAttributes()-1; i++){
				if(!selectedAttsInstance.attribute(i).isString()) {
					double val = selectedAttsInstance.value(selectedAttsInstance.attribute(i));
					featureExtension[i] = val; 
				} 
			}

			String text = null;
			text = instance.stringValue(instance.attribute(1)).toLowerCase();
			text = text.toLowerCase();
			text = text.replace("-", " ").trim();
			text = text.replace("—", " ").trim();
			text = text.replace("classwekaattribute", "class").trim();
			if("".equals(text.trim())){
				continue;
			}
			for(String wrong : wrongWords.keySet()){
				text = (text.replaceAll(wrong, wrongWords.get(wrong).trim())).trim();
			}
			//StringTokenizer tokenizer = new StringTokenizer(text);
			String[] tokens = text.replaceAll("(\\.\\.\\.+|[\\p{Po}\\p{Ps}\\p{Pe}\\p{Pi}\\p{Pf}\u2013\u2014\u2015&&[^'\\.]]|(?<!(\\.|\\.\\p{L}))\\.(?=[\\p{Z}\\p{Pf}\\p{Pe}]|\\Z)|(?<!\\p{L})'(?!\\p{L}))"," $1 ")
					.replaceAll("\\p{C}|^\\p{Z}+|\\p{Z}+$","")
					.split("\\p{Z}+");
			List<double[]> inputWordVectors = new ArrayList<double[]>();
			//while(tokenizer.hasMoreTokens()){
			for(String token : tokens){
				//String token = tokenizer.nextToken().trim();
				if(!token.equals("") && !token.matches(".*lrb.*") && !token.matches(".*rrb.*")) {
					double[] wordVector = null;
					if(tokenVectorMap.containsKey(token)){
						wordVector = tokenVectorMap.get(token);
					} else {
						wordVector = word2Vec.getWordVector(token);
//						if(readSerializedWordVecGoogleModel){
//							wordVector = word2vecGoogle.getWordVector(token);
//						} else if(readGloveVectors){
//							wordVector = GloveVectors.getWordVector(token);
//						} else if(readHLBLVectors){
//							wordVector = HLBLVectors.getWordVector(token);
//						}
					}
					if(wordVector!=null){
						double[] finalFeatureVector = new double[wordVector.length + featureExtension.length];
						int counter = 0;
						for(double w : wordVector){
							finalFeatureVector[counter++] = w;
						}
						for(double w : featureExtension){
							finalFeatureVector[counter++] = w;
						}
						//inputWordVectors.add(finalFeatureVector);
						inputWordVectors.add(wordVector);
						tokenVectorMap.put(token, wordVector);
					} else {
						notFound.add(token);
					}
				}
			}

			double[][] inputSeq = new double[inputWordVectors.size()][];
			inputSeq = inputWordVectors.toArray(inputSeq);
			double[][] targetSeq = new double[inputWordVectors.size()][];
			for(int k=0; k<targetSeq.length; k++) {
				targetSeq[k] = null;//; //make it null to use the error just at last
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
				int count = 0;
				if("Obligation".equalsIgnoreCase(label)){
					count = 2;
				}
				if("Prohibition".equalsIgnoreCase(label)){
					count = 10;
				}
				if("Other".equalsIgnoreCase(label)){
					count = 1;
				}
				for(int j=0; j<count; j++){
					training.add(seq);
				}
				for(int j=0; j<target.length; j++){
					if(target[j] == 1.0){
						actualClassTrainingTotals[j] = actualClassTrainingTotals[j] + count;
					}
				}
			}
		}
		//		for(String token : notFound){
		//			if(!token.matches(".*\\d.*")){
		//				System.out.println(token);
		//			}
		//		}
		System.err.println("\nNot Found" + notFound.size() + "/" + (tokenVectorMap.size() + notFound.size()));
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
			double[] networkOutput = output[output.length-1];
			//			for(int m=0; m<output.length; m++){
			//				for(int k=0; k<output[0].length; k++){
			//					networkOutput[k] = networkOutput[k] + output[m][k]; 
			//				}
			//			}
			//			for(int k=0; k<output[0].length; k++){
			//				networkOutput[k] = networkOutput[k]/output.length;
			//			}
			int winnerIndex = 0;
			double max = Double.MIN_VALUE;
			for(int i=0; i<networkOutput.length; i++){
				if(networkOutput[i]>max){
					winnerIndex = i;
					max = networkOutput[i];
				}
			}
			for(int i=0; i<networkOutput.length; i++){
				if(i!=winnerIndex){
					networkOutput[i] = 0.0;
				} else if(i==winnerIndex){
					networkOutput[i] = 1.0;
				}
			}

			boolean equal = true; 
			totalSteps++;
			for(int i=0; i<networkOutput.length; i++){
				if(networkOutput[i] != actualOutput[i]){
					equal = false; break;
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
			report.append("Predict-count, Actual-count: " + predictedTotalClassTotals[classIndex] + " " + actualClassTestTotals[classIndex] + " \n");
			predictedCorrectClassTotals[classIndex] = 0;
			predictedTotalClassTotals[classIndex] = 0;
		}
		report.append("Test Loss: " + nn.getError() + "\n");
		report.append("Overall Accuracy: " + (int)(correctlyClassified) + "\n");
		nn.resetError();
		return report.toString();
	}

}
