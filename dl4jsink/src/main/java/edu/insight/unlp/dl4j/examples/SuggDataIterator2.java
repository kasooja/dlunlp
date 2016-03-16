package edu.insight.unlp.dl4j.examples;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.Filter;
import weka.filters.MultiFilter;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.StringToWordVector;

/**
 * word embeddings concatenated with external features (weka based)
 * Labels/target: a single class (negative or positive), predicted at the final time step (word) of each review
 *
 * @author K
 */
public class SuggDataIterator2 implements DataSetIterator {
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	private final WordVectors wordVectors;
	private final int batchSize;
	private final int vectorSize;
	private final int truncateLength;

	private int cursor = 0;
	private String dataArffPath;
	private final TokenizerFactory tokenizerFactory;
	private Instances instances;
	private static StringToWordVector stringToWordVectorFilter1;
	private static StringToWordVector stringToWordVectorFilter2;
	private static AttributeSelection attrSel;
	private static MultiFilter multiFilter;
	private boolean train;
	private Instances filteredInstances; 
	public int testCount = 0;

	/**
	 * @param dataDirectory the directory of the IMDB review data set
	 * @param wordVectors WordVectors object
	 * @param batchSize Size of each minibatch for training
	 * @param truncateLength If reviews exceed
	 * @param train If true: return the training data. If false: return the testing data.
	 */
	public SuggDataIterator2(String dataDirectory, WordVectors wordVectors, int batchSize, int truncateLength, boolean train) throws IOException {
		this.batchSize = batchSize;
		this.vectorSize = 400 + wordVectors.lookupTable().layerSize();
		dataArffPath = dataDirectory + "/" + "uv.arff";
		this.train = train;

		if(!train){
			dataArffPath = dataDirectory + "/" + "electronics.arff";
		}

		readDataFile();

		this.wordVectors = wordVectors;
		this.truncateLength = truncateLength;

		tokenizerFactory = new DefaultTokenizerFactory();
		tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
	}

	private void readDataFile() {
		DataSource source;
		try {
			source = new DataSource(dataArffPath);
			instances = source.getDataSet();
			//int k = 0;
//			for(Instance instance : instances){
//				double value = instance.value(instances.attribute("Label")) ;
//				if(value==1.0){
//					k++;
//				} else {
//
//				}
//			}

			//System.out.println(k);
			if(train){
				instances.randomize(instances.getRandomNumberGenerator(799));
				stringToWordVectorFilter1 = Commons.getStringToWordVectorFilter();
				stringToWordVectorFilter2 = Commons.getStringToWordVectorFilter();		
				attrSel = Commons.getAttributeSelectionFilter();		
				NGramTokenizer tok = new NGramTokenizer();
				tok.setNGramMaxSize(1);	
				stringToWordVectorFilter1.setAttributeIndices("2-last");
				stringToWordVectorFilter1.setTokenizer(tok);
				stringToWordVectorFilter1.setAttributeNamePrefix("OtherText_");
				stringToWordVectorFilter2.setAttributeIndices("1");		
				multiFilter = new MultiFilter();
				multiFilter.setFilters(new Filter[]{stringToWordVectorFilter1, stringToWordVectorFilter2, attrSel});				  
			}
			instances.setClassIndex(instances.attribute("Label").index());
			multiFilter.setInputFormat(instances);
			filteredInstances = Filter.useFilter(instances, multiFilter);

//			k = 0;
//			for(Instance instance : filteredInstances){
//				double value = instance.value(filteredInstances.attribute("Label")) ;
//				if(value==1.0){
//					k++;
//				} else {
//
//				}
//			}
//			System.out.println(k);
		} catch(Exception e){
			e.printStackTrace();
		}
	}

	@Override
	public DataSet next(int num) {
		if (cursor >= filteredInstances.numInstances()) throw new NoSuchElementException();
		try{
			return nextDataSet(num);
		}catch(IOException e){
			throw new RuntimeException(e);
		}
	}

	private DataSet nextDataSet(int num) throws IOException {
		//First: load reviews to String. Alternate positive and negative reviews
		boolean[] positive = new boolean[num];
		List<Instance>  reviews = new ArrayList<Instance>();

		for( int i=0; i<num && cursor<totalExamples(); i++ ){
			Instance filtInstance = filteredInstances.get(cursor);
			reviews.add(filtInstance);
			double value = filtInstance.value(filteredInstances.attribute("Label")) ;
			if(value==1.0){
				positive[i] = true;
			} else {
				positive[i] =  false;
			}
			cursor++;
		}

		//		for(boolean l : positive){
		//			if(l){
		//				testCount ++;
		//			}
		//		}

		//Second: tokenize reviews and filter out unknown words
		List<List<String>> allTokens = new ArrayList<>(reviews.size());
		int maxLength = 0;

		int k =0;
		for(Instance instance : reviews){
			Instance origInstance = instances.get(k++);
			String s = origInstance.stringValue(origInstance.attribute(0)).trim();
			List<String> tokens = tokenizerFactory.create(s).getTokens();
			List<String> tokensFiltered = new ArrayList<>();
			for(String t : tokens ){
				if(wordVectors.hasWord(t)) tokensFiltered.add(t);
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
			//			INDArray sumVector =  null;
			//			if(tokens.size()>0){
			//				sumVector = wordVectors.getWordVectorMatrix(tokens.get(0));
			//			} else {
			//				continue;
			//			}
			//			for( int j=1; j<tokens.size() && j<maxLength; j++ ){
			//				String token = tokens.get(j);
			//				INDArray vector = wordVectors.getWordVectorMatrix(token);
			//				sumVector = sumVector.add(vector);
			//			}	
			if(tokens.size()<=0){
				continue;
			}

			Instance filtInstance = reviews.get(i);
			//filtInstance.

			int numAttributes = filtInstance.numAttributes();
			double[] featureExtension = new double[numAttributes-1];	
			for(int m=0; m<filtInstance.numAttributes()-1; m++){
				if(!filtInstance.attribute(m).isString()) {
					double val = filtInstance.value(filtInstance.attribute(m));
					featureExtension[m] = val; 
				} 
			}

			INDArray featExt = Nd4j.create(featureExtension);

			//Get word vectors for each word in review, and put them in the training data
			for( int j=0; j<tokens.size() && j<maxLength; j++ ){
				String token = tokens.get(j);
				INDArray vector = wordVectors.getWordVectorMatrix(token);
				vector = Nd4j.concat(1, featExt, vector);
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
		return filteredInstances.numInstances();
	}

	@Override
	public int inputColumns() {
		return vectorSize;
	}

	@Override
	public int totalOutcomes() {
		return 2;
	}

	@Override
	public void reset() {
		cursor = 0;
		testCount = 0;
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

	//	/** Convenience method for loading review to String */
	//	public String loadReviewToString(int index) throws IOException{
	//		File f;
	//		if(index%2 == 0) f = positiveFiles[index/2];
	//		else f = negativeFiles[index/2];
	//		return FileUtils.readFileToString(f);
	//	}

	//	/** Convenience method to get label for review */
	//	public boolean isPositiveReview(int index){
	//		return index%2 == 0;
	//	}
}