package edu.insight.wordvec.model;


import java.io.File;
import java.io.IOException;

import org.apache.uima.resource.ResourceInitializationException;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.inputsanitation.InputHomogenization;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.util.SerializationUtils;

import edu.insight.unlp.otdf.OTDFXmlReader;


public class Word2VecModelBuilder {

	public static void wikiBuilder(String[] args){		
		OTDFXmlReader otdfXmlReader = new OTDFXmlReader(args[0]);
		SentencePreProcessor sentencePreProcessor = new SentencePreProcessor() {
			private static final long serialVersionUID = 1L;
			public String preProcess(String sentence) {
				sentence = sentence.toLowerCase();
				return new InputHomogenization(sentence).transform();
			}
		};
		SentenceIterator iter = new OTDFSentenceIterator(sentencePreProcessor, otdfXmlReader);		
		try {
			TokenizerFactory t;			
			t = new UimaTokenizerFactory();
			Word2Vec vec = new Word2Vec.Builder().minWordFrequency(40).layerSize(500).windowSize(10).iterate(iter).tokenizerFactory(t).build();			
			vec.fit();
			SerializationUtils.saveObject(vec, new File(args[1]));
		} catch (ResourceInitializationException e1) {
			e1.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args){
		SentencePreProcessor sentencePreProcessor = new SentencePreProcessor() {
			private static final long serialVersionUID = 1L;
			public String preProcess(String sentence) {
				return new InputHomogenization(sentence).transform();
			}
		};
		String electronicsDataFilePath = "/Users/kartik/git/dlunlp/nn/src/main/resources/data/Sequence/electronics.csv";
		//String hotelDataFilePath = "/Users/kartik/git/dlunlp/nn/src/main/resources/data/Sequence/hotel.csv";
		SentenceIterator dataIter = new SuggestionSentenceIterator(sentencePreProcessor, electronicsDataFilePath);
		try {
			TokenizerFactory t;			
			t = new UimaTokenizerFactory();
			Word2Vec vec = new Word2Vec.Builder().minWordFrequency(4).layerSize(10).windowSize(8).iterate(dataIter).tokenizerFactory(t).build();
			vec.fit();
			SerializationUtils.saveObject(vec, new File(args[1]));
		} catch (ResourceInitializationException e1) {
			e1.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}