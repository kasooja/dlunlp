package edu.insight.unlp.dl4j.examples;

import gnu.trove.map.hash.TIntDoubleHashMap;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class GoogleNGramVectors implements Word2Vector {

	private static String gModelPath = "/Users/kartik/Work/dhundo-dobara/Corpus/ML/Corpus/GoogleNews-vectors-negative300.bin.gz";
	private static WordVectors word2Vec;

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

	//	public double getSim(String word1, String word2) {
	//		return word2Vec.similarity(word1, word2);
	//	}

	@Override
	public double getSim(String word1, String word2) {
		//return word2Vec.similarity(word1, word2);
		double[] wordVector1 = word2Vec.getWordVector(word1);
		double[] wordVector2 = word2Vec.getWordVector(word2);
		TIntDoubleHashMap  vector1 = new TIntDoubleHashMap();
		TIntDoubleHashMap  vector2 = new TIntDoubleHashMap();
		for(int i=0; i<wordVector1.length; i++){
			vector1.put(i, wordVector1[i]);
			vector2.put(i, wordVector2[i]);
		}
		return TroveVectorUtils.cosineProduct(vector1, vector2);	
	}

	public static void main(String[] args) {

	}

}
