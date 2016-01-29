package edu.insight.unlp.nn;

import edu.insight.unlp.nn.common.Sequence;

public interface NANN extends NN {

	/*
	 * testing for NA
	 */
	public double[] ff(double[] input, boolean applyTraining);

	/*
	 * testing for NA
	 */
	public double bpNA(double[][] errorGradients, Sequence seq, double learningRate);
	
}
