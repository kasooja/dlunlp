package edu.insight.unlp.dl4j.sugg.acl2016;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.util.SerializationUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import edu.stanford.nlp.ling.TaggedWord;

/** 
 * word vectors concatenated with pos tag
 * Labels/target: a single class (negative or positive), predicted at the final time step (word) of each review
 *
 * @author K
 */
public class WVPOS_SuggDataIterator implements DataSetIterator {

	private static final long serialVersionUID = 1L;
	private final WordVectors wordVectors;
	private final int batchSize;
	private final int vectorSize;
	private final int truncateLength;

	private int cursor = 0;

	private final List<String> positiveFiles;
	private final List<String> negativeFiles;

	private static Map<String, List<TaggedWord>> tagMap = new HashMap<String, List<TaggedWord>>();
	private static String tagMapPath = "src/main/resources/suggData/sentTag.map"; 

	static {
		if(new File(tagMapPath).exists()){
			tagMap = SerializationUtils.readObject(new File(tagMapPath));
		}
	}

	public WVPOS_SuggDataIterator(WordVectors wordVectors, List<String> positiveFiles, List<String> negativeFiles, int batchSize, int truncateLength){
		this.positiveFiles = positiveFiles;
		this.negativeFiles = negativeFiles;
		this.batchSize = batchSize;
		this.vectorSize = 1 + wordVectors.lookupTable().layerSize();
		this.wordVectors = wordVectors;
		this.truncateLength = truncateLength;
	}

	@Override
	public DataSet next(int num) {
		if (cursor >= positiveFiles.size() + negativeFiles.size()) throw new NoSuchElementException();
		try{
			return nextDataSet(num);
		}catch(IOException e){
			throw new RuntimeException(e);
		}
	}

	private DataSet nextDataSet(int num) throws IOException {
		//First: load reviews to String. Alternate positive and negative reviews
		List<String> reviews = new ArrayList<String>();
		boolean[] positive = new boolean[num];
		for( int i=0; i<num && cursor<totalExamples(); i++ ){
			if(cursor % 2 == 0){
				//Load positive review
				int posReviewNumber = cursor / 2;
				if(posReviewNumber<positiveFiles.size()){
					String review = positiveFiles.get(posReviewNumber).trim();
					reviews.add(review);
					positive[i] = true;
				}
			} else {
				//Load negative review
				int negReviewNumber = cursor / 2;		
				if(negReviewNumber<negativeFiles.size()){
					String review = negativeFiles.get(negReviewNumber).trim();
					reviews.add(review);
					positive[i] = false;
				}
			}
			cursor++;
		}

		//Second: tokenize reviews and filter out unknown words
		List<List<String>> allTokens = new ArrayList<List<String>>();
		int maxLength = 0;
	
		for(String s : reviews){
			if(!tagMap.containsKey(s)){
				List<TaggedWord> tagText = StanfordNLPUtil.getTagText(s);
				tagMap.put(s,  tagText);					
			}
			List<TaggedWord> tokens = tagMap.get(s);
	
			List<String> tokensFiltered = new ArrayList<>();
			for(TaggedWord t : tokens){
				if(wordVectors.hasWord(t.word().toLowerCase())) {
					tokensFiltered.add(t.word().toLowerCase()+ "specialCharSpecialchar" + t.tag());
				}
			}
			allTokens.add(tokensFiltered);
			maxLength = Math.max(maxLength,tokensFiltered.size());
		}
		//If longest review exceeds 'truncateLength': only take the first 'truncateLength' words
		if(maxLength > truncateLength) maxLength = truncateLength;

		//Create data for training
		//Here: we have reviews.size() examples of varying lengths
		INDArray features = Nd4j.create(reviews.size(), vectorSize, maxLength);
		INDArray labels = Nd4j.create(reviews.size(), 2, maxLength);    //Two labels: positive or negative
		//Because we are dealing with reviews of different lengths and only one output at the final time step: use padding arrays
		//Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
		INDArray featuresMask = Nd4j.zeros(reviews.size(), maxLength);
		INDArray labelsMask = Nd4j.zeros(reviews.size(), maxLength);

		int[] temp = new int[2];
		for( int i=0; i<reviews.size(); i++ ){
			List<String> tokens = allTokens.get(i);
			temp[0] = i;

			//Get word vectors for each word in review, and put them in the training data
			for( int j=0; j<tokens.size() && j<maxLength; j++ ){
				String[] split = tokens.get(j).split("specialCharSpecialchar");
				String token = split[0].trim();
				String tag = split[1].trim();
				int tagIndex = StanfordNLPUtil.stanTags.indexOf(tag);
				double[] tagIndexArray = new double[1];
				tagIndexArray[0] = tagIndex;
				INDArray tagIndexVector = Nd4j.create(tagIndexArray);
				INDArray vector = wordVectors.getWordVectorMatrix(token);
				vector = Nd4j.concat(1, tagIndexVector, vector);
				features.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j)}, vector);
				temp[1] = j;
				featuresMask.putScalar(temp, 1.0);  //Word is present (not padding) for this example + time step -> 1.0 in features mask
			}

			int idx = (positive[i] ? 0 : 1);
			int lastIdx = Math.min(tokens.size(),maxLength);
			labels.putScalar(new int[]{i,idx,lastIdx-1},1.0);   //Set label: [0,1] for negative, [1,0] for positive
			labelsMask.putScalar(new int[]{i,lastIdx-1},1.0);   //Specify that an output exists at the final time step for this example
		}

		return new DataSet(features,labels,featuresMask,labelsMask);
	}

	@Override
	public int totalExamples() {
		return positiveFiles.size() + negativeFiles.size();
	}

	@Override
	public int inputColumns() {
		return vectorSize;
	}

	@Override
	public int totalOutcomes() {
		return 2;
	}

	//	public void serialize(){
	//		SerializationUtils.saveObject(tagMap, new File(tagMapPath));
	//	}

	@Override
	public void reset() {		
		cursor = 0;
	}

	@Override
	public int batch() {
		return batchSize;
	}

	@Override
	public int cursor() {
		return cursor;
	}

	@Override
	public int numExamples() {
		return totalExamples();
	}

	@Override
	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		throw new UnsupportedOperationException();
	}

	@Override
	public List<String> getLabels() {
		return Arrays.asList("positive","negative");
	}

	@Override
	public boolean hasNext() {
		return cursor < numExamples();
	}

	@Override
	public DataSet next() {
		return next(batchSize);
	}

	@Override
	public void remove() {

	}
	
}