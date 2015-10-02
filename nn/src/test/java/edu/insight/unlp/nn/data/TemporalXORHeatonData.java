/*
 * Encog(tm) Java Examples v3.3
 * http://www.heatonresearch.com/encog/
 * https://github.com/encog/encog-java-examples
 *
 * Copyright 2008-2014 Heaton Research, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *   
 * For more information on Heaton Research copyrights, licenses 
 * and trademarks visit:
 * http://www.heatonresearch.com/copyright
 */
package edu.insight.unlp.nn.data;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import edu.insight.unlp.nn.DataSet;
import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.common.Sequence;

/**
 * Utility class that presents the XOR operator as a serial stream of values.
 * This is used to predict the next value in the XOR sequence. This provides a
 * simple stream of numbers that can be predicted.
 * 
 * @author jeff
 * 
 */
public class TemporalXORHeatonData extends DataSet {

	/**
	 * 1 xor 0 = 1, 0 xor 0 = 0, 0 xor 1 = 1, 1 xor 1 = 0
	 */
	public static final double[] SEQUENCE = { 1.0, 0.0, 1.0, 0.0, 0.0, 0.0,
		0.0, 1.0, 1.0, 1.0, 1.0, 0.0 };
	private static Random rng = new Random();
	private static int sequenceMinLength = 4;
	private static int sequenceMaxLength = 7;

	private static int trainingSeqs = 1000;
	private static int testingSeqs = 100;

	public TemporalXORHeatonData(){
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

	public static List<Sequence> getSequences(int count) {
		List<Sequence> seqs = new ArrayList<Sequence>();
		for(int i=0; i<count; i++){
			int len = rng.nextInt(sequenceMaxLength - sequenceMinLength + 1) + sequenceMinLength;
			double[][] input;
			double[][] ideal;
			input = new double[len][1];
			ideal = new double[len][1];
			for (int j = 0; j < input.length; j++) {
				input[j][0] = TemporalXORHeatonData.SEQUENCE[j
				                                             % TemporalXORHeatonData.SEQUENCE.length];
				ideal[j][0] = TemporalXORHeatonData.SEQUENCE[(j + 1)
				                                             % TemporalXORHeatonData.SEQUENCE.length];
			}
			Sequence seq = new Sequence(input, ideal);
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