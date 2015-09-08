
package edu.insight.unlp.nn;

import java.util.List;


public interface NN {

	public double[] output(double[] input);
	
	public double[] outputSequence(double[][] inputSeq);

	public void update(double learningRate);

	public int numOutputUnits();
	
	public void initializeNN();

	public double error(double[] inputs, double[] target);

	public double batchgdTrain(double[][] inputs, double[][] targets, double learningRate, int batchSize, boolean shuffle);

	/*
	 * sgd not require, as batchgd function can do it, by giving the batch size = 1, and shuffle true
	 */
	public double sgdTrain(double[][] inputs, double[][] targets, double learningRate);
	
	public void setLayers(List<NNLayer> layers);
	
	public List<NNLayer> getLayers();
	
	public void resetActivationCounter();
	

}