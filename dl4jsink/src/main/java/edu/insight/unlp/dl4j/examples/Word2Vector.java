package edu.insight.unlp.dl4j.examples;

public interface Word2Vector {
	
	public double[] getWordVector(String word);

	public double getSim(String word1, String word2);

}
