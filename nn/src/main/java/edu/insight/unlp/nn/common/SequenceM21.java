package edu.insight.unlp.nn.common;

/*
 * many to one (M21) sequence
 */
public class SequenceM21 {
	
	public double[][] inputSeq;
	public double[] target;
	
	public SequenceM21(double[][] inputSeq, double[] target){
		this.inputSeq = inputSeq;
		this.target = target;
	}

}
