package edu.insight.unlp.nn.utils;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import edu.insight.unlp.nn.common.Sequence;


public class TemporalXOR {

	/**
	 * 1 xor 0 = 1, 0 xor 0 = 0, 0 xor 1 = 1, 1 xor 1 = 0
	 */
	public static final double[] SEQUENCE = { 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0 };
	//private static final Map<String, Double> xorOutput = new HashMap<String, Double>();

	private static Random rng = new Random();
	private static int sequenceMinLength = 7;
	private static int sequenceMaxLength = 11;

	public static Double getXorOutput(double first, double second){
		if(Math.round(first) + Math.round(second) == 2.0 || Math.round(first) + Math.round(second) == 0.0){
			return 0.0;
		}
		if(Math.round(first) + Math.round(second) == 1.0){
			return 1.0;
		} 
		return null;
	}

	public static List<Sequence> generate(int count) {
		List<Sequence> seqs = new ArrayList<Sequence>();
		boolean firstSeq = false;
		double[] constantInput = new double[]{1, 0};
		for(int i=0; i<count; i++){
			int len = rng.nextInt(sequenceMaxLength - sequenceMinLength + 1) + sequenceMinLength;
			if(firstSeq){
				len = constantInput.length;				
			} 
			int constantCounter = 0;
			double[][] inputSeq = new double[len][1];
			double[][] outputSeq = new double[len][1];
			double prevXorInput = 0.0;
			for(int j=0; j<len; j++){
				double randomXorInput = (double) Math.round(Math.random());
				if(firstSeq){
					randomXorInput = constantInput[constantCounter++];
				}
				Double xorOutput = getXorOutput(prevXorInput, randomXorInput);
				if(xorOutput == null){
					System.err.println("ERRRRROR");
				}
				inputSeq[j][0] = randomXorInput;
				outputSeq[j][0] = xorOutput;
				prevXorInput = randomXorInput;
			}
			if(firstSeq){
				firstSeq = false;
			}
			Sequence seq = new Sequence(inputSeq, outputSeq);
			seqs.add(seq);
		}
		return seqs;
	}

	public static void main(String[] args) {
		//TemporalXOR d = new TemporalXOR();
		//d.generate(5);
		System.out.println("k");
	}

}