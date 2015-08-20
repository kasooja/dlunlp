package edu.insight.unlp.word2vec;


import java.io.File;

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.util.SerializationUtils;

public class Word2VecTest {

	public static void test(Word2Vec vec){
		String one = "sex";
		String two = "like";
		System.out.println(vec.wordsNearest(one, 100));//gsimilarWordsInVocabTo("love", 0.8));
		System.out.println(vec.similarity(one, two));
		System.out.println(vec.similarity(one, "inout"));		
		System.out.println(vec.similarity(one, "fuck"));
		
		System.out.println();
	}

	public static void main(String[] args) {
		Word2Vec vec = SerializationUtils.readObject(new File(args[0]));
		System.out.println("loading done");
		test(vec);
		System.out.println("exp done");		
	}

}
