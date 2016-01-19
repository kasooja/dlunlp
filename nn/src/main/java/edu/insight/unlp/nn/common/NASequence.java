package edu.insight.unlp.nn.common;

public class NASequence extends Sequence{

	public double[][] naInput;
	
	public NASequence(double[][] inputSeq, double[][] target, double[][] naInput) {
		super(inputSeq, target);
		this.naInput = naInput;		
	}

}
