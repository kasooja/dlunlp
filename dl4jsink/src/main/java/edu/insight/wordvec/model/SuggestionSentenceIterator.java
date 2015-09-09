package edu.insight.wordvec.model;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

import org.deeplearning4j.text.sentenceiterator.BaseSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;

import au.com.bytecode.opencsv.CSVReader;

public class SuggestionSentenceIterator extends BaseSentenceIterator {

	private String dataFilePath;
	private CSVReader reader;
	private String currentSentence;

	public SuggestionSentenceIterator(SentencePreProcessor preProcessor, String dataFilePath){
		super(preProcessor);
		this.dataFilePath = dataFilePath;
		setIterator();
	}

	private void setIterator(){
		try {
			reader = new CSVReader(new InputStreamReader(new FileInputStream(dataFilePath), "UTF-8"));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@Override
	public boolean hasNext() {
		try {
			String[] values = reader.readNext();
			if(values != null ) {
				currentSentence = values[1];
				return true;
			} else {
				reader.close();
			}
		} catch (IOException e1) {
			e1.printStackTrace();
		}
		return false;
	}

	@Override
	public String nextSentence() {
		return currentSentence;
	}

	@Override
	public void reset() {
		setIterator();	
	}

}
