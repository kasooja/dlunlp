package edu.insight.wordvec.model;

import java.io.IOException;
import java.util.Iterator;

import org.deeplearning4j.text.sentenceiterator.BaseSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;

import edu.insight.unlp.otdf.OTDFFile;
import edu.insight.unlp.otdf.OTDFXmlReader;


public class OTDFSentenceIterator extends BaseSentenceIterator {

	public enum WikiOTDFTags {
		Title, URI_EN, Article, LanguageISOCode;	
	}

	private OTDFXmlReader otdfXmlReader;
	private Iterator<OTDFFile> iterator = null;

	public OTDFSentenceIterator(SentencePreProcessor preProcessor, OTDFXmlReader otdfXmlReader){
		super(preProcessor);
		this.otdfXmlReader = otdfXmlReader;
		setIterator();
	}

	private void setIterator(){		
		try {
			iterator = otdfXmlReader.getIterator();
		} catch (IOException e) {
			e.printStackTrace();
		}		
	}

	public String nextSentence() {	
		OTDFFile file = iterator.next();				
		file.getFeatureValue(WikiOTDFTags.URI_EN.toString());							
		file.getFeatureValue(WikiOTDFTags.Title.toString());				
		String sentenceString = file.getFeatureValue(WikiOTDFTags.Article.toString());
		return sentenceString;
	}

	public boolean hasNext() {
		boolean hasNext = iterator.hasNext();
		if(hasNext){
		} else {
			otdfXmlReader.close();			
		}		
		return hasNext;
	}

	public void reset() {

	}

}