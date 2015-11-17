package edu.insight.unlp.word2vec;

import java.io.File;
import java.io.IOException;

import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.Word2Vec;

public class Word2VecLoaderTest {
	
	public static void test(WordVectors vec){
		String one = "sex";
		String two = "like";
		System.out.println(vec.wordsNearest(one, 100));//gsimilarWordsInVocabTo("love", 0.8));
		System.out.println(vec.similarity(one, two));
		System.out.println(vec.similarity(one, "inout"));		
		System.out.println(vec.similarity(one, "fuck"));
		
		System.out.println();
	}

	
	public static void main(String[] args) {
		File gModel = new File("/Users/kartik/Work/dhundo-dobara/Corpus/ML/Corpus/GoogleNews-vectors-negative300.bin.gz");
		try {
			WordVectors vec = WordVectorSerializer.loadGoogleModel(gModel, true);
			System.out.println("loading done");
			test(vec);
			System.out.println("exp done");		
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
