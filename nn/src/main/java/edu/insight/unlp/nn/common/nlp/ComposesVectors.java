package edu.insight.unlp.nn.common.nlp;
import java.io.BufferedReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import edu.insight.unlp.nn.utils.BasicFileTools;

public class ComposesVectors implements Word2Vector {

	public Map<String, double[]> vecs = new HashMap<String, double[]>();
	public String filePath = "/Users/kartik/Downloads/John/STS/Publications/Distributed:Distributional Vectors/ Don't count, predict! A systematic comparison of context-counting vs. context-predicting semantic vectors /Data/EN-wform.w.5.cbow.neg10.400.subsmpl.txt";

	/*
	 * provide the embeddings filePath 
	 */
	public ComposesVectors(String filePath) {
		this.filePath = filePath;
		loadEmbeddings(this.filePath);
	}
	
	/*
	 * Uses the default path for embeddings
	 */
	public ComposesVectors() {
		loadEmbeddings(filePath);
	}
	
	private void loadEmbeddings(String filePath){
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

	public double[] getWordVector(String word){
		double[] vec = vecs.get(word);
		if(vec==null){
			vec = vecs.get("unknowns");
		}
		return vec;
	}

	public static void main(String[] args) {
		ComposesVectors gvec = new ComposesVectors();
		double[] ds = gvec.getWordVector("the");
		for(double j : ds){
			System.out.println(j);
		}
	}

	@Override
	public double getSim(String word1, String word2) {
		if(word1.equals(word2))
			return 1.0;
		INDArray array1 = Nd4j.create(getWordVector(word1));
		INDArray array2 = Nd4j.create(getWordVector(word2));
		INDArray vector1 = Transforms.unitVec(array1);
		INDArray vector2 = Transforms.unitVec(array2);
		if(vector1 == null || vector2 == null)
			return -1;
		return  Nd4j.getBlasWrapper().dot(vector1, vector2);
	}

}


