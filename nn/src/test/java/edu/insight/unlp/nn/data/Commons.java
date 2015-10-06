package edu.insight.unlp.nn.data;


import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.Random;

import edu.insight.unlp.nn.utils.BasicFileTools;
import weka.classifiers.Classifier;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.Utils;
import weka.core.tokenizers.NGramTokenizer;
import weka.filters.supervised.attribute.AttributeSelection;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.RemoveByName;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class Commons {

	/**
	 * Loads the dataset from disk.
	 * 
	 * @param file the dataset to load (e.g., "weka/classifiers/data/something.arff")
	 * @throws Exception if loading fails, e.g., file does not exit
	 */
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

	public static AttributeSelection getAttributeSelectionFilter() {	
		AttributeSelection attrSel = new AttributeSelection();
		String[] options = new String[4];
		options[0] = "-E";
		options[1] = "weka.attributeSelection.InfoGainAttributeEval -B";
		options[2] = "-S";	
		options[3] = "weka.attributeSelection.Ranker -T -1.7976931348623157E308 -N 500";
		try {
			attrSel.setOptions(options);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return attrSel;
	}

	
	public static Remove getRemoveFilter(String index) {	
		Remove remove = new Remove();		
		String[] options = new String[2];
		options[0] = "-R";
		options[1] = index;
		try {
			remove.setOptions(options);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return remove;
	}
	
	public static RemoveByName getRemoveFilterByRegex(String regex) {	
		RemoveByName remove = new RemoveByName();		
		String[] options = new String[2];
		options[0] = "-E";
		options[1] = regex;
		try {
			remove.setOptions(options);
		} catch (Exception e) {
			e.printStackTrace();
		}
		return remove;
	}

	public static StringToWordVector getStringToWordVectorFilter() {		
		SelectedTag selectedTag = new SelectedTag(StringToWordVector.FILTER_NORMALIZE_ALL, StringToWordVector.TAGS_FILTER);
		StringToWordVector stringToWordVector = new StringToWordVector();	
		stringToWordVector.setWordsToKeep(1000);
		stringToWordVector.setTFTransform(true);
		stringToWordVector.setIDFTransform(true);	
		stringToWordVector.setNormalizeDocLength(selectedTag);		
		stringToWordVector.setMinTermFreq(4);
		stringToWordVector.setLowerCaseTokens(true);
		stringToWordVector.setDoNotOperateOnPerClassBasis(false);
		NGramTokenizer tok = new NGramTokenizer();
		tok.setNGramMaxSize(2);
		stringToWordVector.setTokenizer(tok);
		stringToWordVector.setOutputWordCounts(true);
		//stringToWordVector.setUseStoplist(true);
		return stringToWordVector;
	}


}
