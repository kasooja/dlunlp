package edu.insight.unlp.dl4j.examples;

import java.io.BufferedReader;
import java.io.IOException;

import edu.insight.unlp.otdf.BasicFileTools;

public class EntityRelatednessEval {

	private static Word2Vector word2vec = new GoogleNGramVectors();
	//private static Word2Vector word2vec = new GloveVectors();

	public static void main(String[] args) {
		String filePath = "src/main/resources/entityDataset/KOREEntitiesPairs.txt";
		BufferedReader br = BasicFileTools.getBufferedReader(filePath);
		String line = null;
		StringBuilder bld = new StringBuilder();
		try {
			int i=0;
			while((line=br.readLine())!=null){
				System.out.println(i++);
				String[] split = line.split("\\t");
				String entity1 = split[0];//.replaceAll("\\s+", "_");
				String entity2 = split[1];//.replaceAll("\\s+", "_");
				Double simScore = null;
				double simTest =  word2vec.getSim("love", "like");
				System.out.println("love like " + simTest);
				try {
					double sim =  word2vec.getSim(entity1.toLowerCase().trim(), entity2.toLowerCase().trim());
					simScore = sim;
				} catch(Exception e){
					System.err.println("not found");
				}
				if(simScore == null){
					bld.append(split[0] + "\t" + split[1] + "\t" + "NotFound" + "\n");
				} else {
					bld.append(split[0] + "\t" + split[1] + "\t" + String.valueOf(simScore)  + "\n");
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		BasicFileTools.writeFile("src/main/resources/entityDataset/result_word2vec.tsv", bld.toString().trim());
	}

}
