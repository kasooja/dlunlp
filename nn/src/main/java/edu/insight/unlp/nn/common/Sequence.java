package edu.insight.unlp.nn.common;

/*
 * many to many sequence
 */
public class Sequence {
	
	public double[][] inputSeq;
	public double[][] target;
	public String labelString;
	
	
	public Sequence(double[][] inputSeq, double[][] target){
		this.inputSeq = inputSeq;
		this.target = target;
	}

}
