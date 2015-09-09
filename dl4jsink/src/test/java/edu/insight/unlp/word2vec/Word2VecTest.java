package edu.insight.unlp.word2vec;


import java.io.File;

import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.util.SerializationUtils;

public class Word2VecTest {

	public static void test(Word2Vec vec){
		String one = "apex";
		String two = "dvd";
		System.out.println(vec.wordsNearest(one, 100));;
		System.out.println(vec.similarity(one, two));
		double[] wordVector = vec.getWordVector(two);
		System.out.println(wordVector.length);
		for(double s : wordVector){
			System.out.println(s);
			
		}
	}

	public static void main(String[] args) {
		Word2Vec vec = SerializationUtils.readObject(new File(args[0]));
		System.out.println("loading done");
		test(vec);
		System.out.println("exp done");		
	}
	
	
	

}
