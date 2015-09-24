package edu.insight.unlp.nn;


public interface MLP extends NN {

	public double sgdTrain(double[][] inputs, double[][] targets, double learningRate, boolean shuffle);

	public double[] output(double[] input);

}
