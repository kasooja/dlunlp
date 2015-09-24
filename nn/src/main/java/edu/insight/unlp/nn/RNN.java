package edu.insight.unlp.nn;

import java.util.List;

import edu.insight.unlp.nn.common.Sequence;

public interface RNN extends NN{
	
	public double sgdTrain(List<Sequence> training, double learningRate, boolean shuffle);
	
	public double[][] output(double[][] inputSeq);

}
