package edu.insight.unlp.nn.data;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import edu.insight.unlp.nn.DataSet;
import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.common.Sequence;


public class TemporalXORData extends DataSet {

	/**
	 * 1 xor 0 = 1, 0 xor 0 = 0, 0 xor 1 = 1, 1 xor 1 = 0
	 */
	public static final double[] SEQUENCE = { 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0 };
	private static Random rng = new Random();
	private static int sequenceMinLength = 7;
	private static int sequenceMaxLength = 11;
	private static int trainingSeqs = 1000;
	private static int testingSeqs = 100;

	public TemporalXORData(){
		setDataSet();
	}
	
	public void setDataSet(){
		this.training = getSequences(trainingSeqs);
		this.testing = getSequences(testingSeqs);
		inputUnits = training.get(0).inputSeq[0].length;
		for(Sequence seq : training) {
			if(seq.target!=null){		
				outputUnits = seq.target[0].length;
				break;
			}
		}
	}

	public static Double getXorOutput(double first, double second){
		if(Math.round(first) + Math.round(second) == 2.0 || Math.round(first) + Math.round(second) == 0.0){
			return 0.0;
		}
		if(Math.round(first) + Math.round(second) == 1.0){
			return 1.0;
		} 
		return null;
	}

	private static List<Sequence> getSequences(int count) {
		List<Sequence> seqs = new ArrayList<Sequence>();
		boolean firstSeq = true;
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
					System.err.println("Error in producing xorOutput");
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

	@Override
	public String evaluateTest(NN nn) {
		int totalCorrect = 0;
		int totalSteps = 0;
		StringBuilder report = new StringBuilder();
		for(Sequence seq : testing) {
			double[][] inputSeq = seq.inputSeq;
			double[][] target = seq.target;
			double[][] output = nn.output(inputSeq);
			totalSteps = totalSteps +  inputSeq.length;
			for(double[] out : output){
				out[0] = Math.round(out[0]);
			}
			for(int i=0; i<output.length; i++){
				if(target[i][0] != output[i][0]){
				} else {
					totalCorrect++;
				}
			}
			nn.resetActivationCounter(false);				
		}			
		double correctlyClassified = ((double)totalCorrect/(double)totalSteps) * 100;
		report.append((int)correctlyClassified + "% correctly classified");
		return report.toString();
	}
	
}

