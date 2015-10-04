package edu.insight.unlp.nn.common.nlp;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import edu.insight.unlp.nn.utils.BasicFileTools;

public class GloveVectors {

	public static Map<String, double[]> vecs = new HashMap<String, double[]>();

	static {
		String filePath = "src/test/resources/glove.6B.50d.txt";
		BufferedReader br = BasicFileTools.getBufferedReader(filePath);
		String line = null;
		try {
			while((line=br.readLine())!=null){
				String[] split = line.split("\\s+");
				String word = split[0].trim();
				int i=1;
				double[] vec = new double[split.length - 1];
				for(i=1; i<split.length; i++){
					vec[i-1] = Double.parseDouble(split[i]);
				}
				vecs.put(word, vec);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
	
	public static void main(String[] args) {
		double[] ds = vecs.get("the");
		for(double j : ds){
			System.out.println(j);
		}
	}
}


