package edu.insight.unlp.nn.common.nlp;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;

public class GoogleNGramVectors implements Word2Vector {

	private static String gModelPath = "/Users/kartik/Work/dhundo-dobara/Corpus/ML/Corpus/GoogleNews-vectors-negative300.bin.gz";
	private static Word2Vec word2Vec;

	static {
		try {
			System.err.print("Loading GoogleNews-vectors-negative300 Word2Vec model . . .");
			word2Vec = WordVectorSerializer.loadGoogleModel(new File(gModelPath), true);
			System.err.println("Done");
		} catch (IOException e) {
			e.printStackTrace();
		}	
	}

	@Override
	public double[] getWordVector(String word) {
		return word2Vec.getWordVector(word);
	}

}
