package edu.insight.unlp.nn.common;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import edu.insight.unlp.nn.utils.BasicFileTools;

public class HLBLVectors {

	public static Map<String, double[]> vecs = new HashMap<String, double[]>();

	static {
		String filePath = "src/test/resources/hlbl-embeddings-scaled.EMBEDDING_SIZE=50.txt";
		BufferedReader br = BasicFileTools.getBufferedReaderFile(filePath);
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
}


