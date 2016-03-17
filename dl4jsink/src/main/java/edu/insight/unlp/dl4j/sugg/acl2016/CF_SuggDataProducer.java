package edu.insight.unlp.dl4j.sugg.acl2016;

import java.io.File;
import java.io.FileNotFoundException;
import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.util.SerializationUtils;

import au.com.bytecode.opencsv.CSV;
import au.com.bytecode.opencsv.CSVReadProc;
import edu.stanford.nlp.ling.TaggedWord;

public class CF_SuggDataProducer {

	private int posCursor = 0;
	private int negCursor = 0;
	private int negFoldSize = 0;
	private int posFoldSize = 0;
	private int noOfFolds = 0;
	private String dataPath = null; //.csv file
	private WordVectors wordVectors;
	private int trainBatchSize;
	private int testBatchSize;
	private int truncateReviewsToLength;
	private int csvTotCount = 0;
	private int invalidCsvRows = 0;
	private List<String> pos = new ArrayList<String>();
	private List<String> neg = new ArrayList<String>();
	private Constructor<? extends DataSetIterator> dataItConstructor;

	///*
//	private static Map<String, List<TaggedWord>> tagMap = new HashMap<String, List<TaggedWord>>();
//	private static String tagMapPath = "src/main/resources/suggData/sentTag.map"; 
//
//	private int howManyNot = 0;
//
//	static {
//		if(new File(tagMapPath).exists()){
//			tagMap = SerializationUtils.readObject(new File(tagMapPath));
//		}
//	}
	//*/

	public CF_SuggDataProducer(int noOfFolds, String dataPath, WordVectors wordVectors, int trainBatchSize,
			int testBatchSize, int truncateReviewsToLength, Class<? extends DataSetIterator> dataSetIterator) {
		this.noOfFolds = noOfFolds;
		this.dataPath = dataPath;
		this.wordVectors = wordVectors;
		this.trainBatchSize = trainBatchSize;
		this.testBatchSize = testBatchSize;
		this.truncateReviewsToLength = truncateReviewsToLength;		
		dataItConstructor = getDataItConstructor(dataSetIterator);
		loadData();
	}

	private Constructor<? extends DataSetIterator> getDataItConstructor(Class<? extends DataSetIterator> dataSetIterator){
		try {
			return dataSetIterator.getConstructor(wordVectors.getClass(), pos.getClass(), neg.getClass(), int.class, int.class);
		} catch (NoSuchMethodException e) {
			e.printStackTrace();
		} catch (SecurityException e) {
			e.printStackTrace();
		}
		return null;
	}

	public void loadData(){
		CSV csv = CSV.create();
		csv.read(dataPath, new CSVReadProc() {
			public void procRow(int rowIndex, String... values) {
				
				///*
			//	System.out.println(csvTotCount++);
				//*/
				
				String id = values[0];
				if("id".equals(id)){
					return;
				}

				String text = values[1].trim();
				int suggLabel = Integer.parseInt(values[2].trim());

				if("text".equals(text) || "".equals(text) || (suggLabel != 0 && suggLabel != 1)){
					invalidCsvRows++;
					return;
				}			

				///*
//				if(!tagMap.containsKey(text)){	
//					howManyNot++;
//					List<TaggedWord> tagText = StanfordNLPUtil.getTagText(text);
//					tagMap.put(text,  tagText);					
//				}
				//*/

				if(suggLabel == 0){
					neg.add(text);					
				} else if(suggLabel == 1){
					pos.add(text);
				}

			}			
		});

		long seed = System.nanoTime();

		Collections.shuffle(neg, new Random(seed));
		Collections.shuffle(pos, new Random(seed));

		negFoldSize = neg.size()/noOfFolds;
		posFoldSize = pos.size()/noOfFolds;

		System.out.println("Total Rows Count: " +  csvTotCount);
		System.out.println("invalid Rows Count: " +  invalidCsvRows);
		System.out.println("Total Pos Examples: " + pos.size());
		System.out.println("Total Neg Examples: " + neg.size());		
	}

	public Pair<DataSetIterator, DataSetIterator> nextPair(){
		List<String> testNegs = new ArrayList<String>();
		List<String> testPoss = new ArrayList<String>();

		for(int i=0; i<negFoldSize && negCursor<neg.size(); i++){
			String negEx = neg.get(negCursor);
			testNegs.add(negEx);	
			negCursor++;
		}

		for(int i=0; i<posFoldSize && posCursor<pos.size(); i++){
			String posEx = pos.get(posCursor);
			testPoss.add(posEx);
			posCursor++;
		}

		DataSetIterator testIt = getNewDataItrInstance(testPoss, testNegs, testBatchSize);
		
		List<String> trainNegs = new ArrayList<String>();
		List<String> trainPoss = new ArrayList<String>();

		for(String negEx : neg){
			if(!testNegs.contains(negEx)){
				trainNegs.add(negEx);
			}
		}

		for(String posEx : pos){
			if(!testPoss.contains(posEx)){
				trainPoss.add(posEx);
			}
		}

		Collections.shuffle(trainNegs);
		Collections.shuffle(trainPoss);

		if(trainNegs.size()==trainPoss.size()){
		} else if(trainNegs.size()<trainPoss.size()){
			int diff = trainPoss.size() - trainNegs.size();
			int origSize = trainNegs.size();
			for(int k=0; k<diff; k++){
				int[] randArray = new Random().ints(1, 0, origSize).toArray();
				int indexToBeAdded = randArray[0];
				trainNegs.add(trainNegs.get(indexToBeAdded));
			}
		} else if(trainPoss.size()<trainNegs.size()){
			int diff = trainNegs.size() - trainPoss.size();
			int origSize = trainPoss.size();
			for(int k=0; k<diff; k++){
				int[] randArray = new Random().ints(1, 0, origSize).toArray();
				int indexToBeAdded = randArray[0];
				trainPoss.add(trainPoss.get(indexToBeAdded));
			}
		}		

		System.out.println(trainPoss.size()==trainNegs.size());
	
		DataSetIterator trainIt = getNewDataItrInstance(trainPoss, trainNegs, trainBatchSize);	

		Pair<DataSetIterator, DataSetIterator> pair = new Pair<DataSetIterator, DataSetIterator>(trainIt, testIt);
		return pair;
	}

	private DataSetIterator getNewDataItrInstance(List<String> poss, List<String> negs, int batchSize){
		try {
			DataSetIterator newInstance = dataItConstructor.newInstance(wordVectors, poss, negs, batchSize, truncateReviewsToLength);
			return newInstance;
		} catch (InstantiationException e) {
			e.printStackTrace();
		} catch (IllegalAccessException e) {
			e.printStackTrace();
		} catch (IllegalArgumentException e) {
			e.printStackTrace();
		} catch (InvocationTargetException e) {
			e.printStackTrace();
		}
		return null;
	}
	
//	public static void serialize(){
//		SerializationUtils.saveObject(tagMap, new File(tagMapPath));
//	}

	public static void main(String[] args) {
		String DATA_PATH = "/home/kartik/git/dlunlp/dl4jsink/src/main/resources/suggData/hotel.csv";
		String WORD_VECTORS_PATH_GLOVE = "/home/kartik/git/dlunlp/dl4jsink/src/main/resources/models/glove.6B.50d.txt";
		WordVectors wordVectors = null;
		try {
			wordVectors = Dl4j_ExtendedWVSerializer.loadTxtVectors(new File(WORD_VECTORS_PATH_GLOVE));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		int trainBatchSize = 10;
		int testBatchSize = 100;
		int truncateReviewsToLength = 5;
		CF_SuggDataProducer dataProducer = new CF_SuggDataProducer(10, DATA_PATH, wordVectors, 
				trainBatchSize, testBatchSize, truncateReviewsToLength, WVPOS_SuggDataIterator.class);
		System.out.println("Total Data Size: " + dataProducer.csvTotCount);
		//System.out.println("Map not contains: " + dataProducer.howManyNot + " items");
		//System.out.println("Map: " + tagMap.size());
		//serialize();
	}

}
