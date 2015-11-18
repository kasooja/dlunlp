package edu.insight.unlp.nn.common.nlp;

public interface Word2Vector {

	public double[] getWordVector(String word);

	public double getSim(String word1, String word2);
	
	public void setEmbeddingFilePath(String filePath);

}
