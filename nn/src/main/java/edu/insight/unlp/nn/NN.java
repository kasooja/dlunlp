
package edu.insight.unlp.nn;

import java.util.List;

import edu.insight.unlp.nn.common.Sequence;


public interface NN {
	
	public void update(double learningRate);

	public int numOutputUnits();
	
	public void initializeNN();
	
	public void setLayers(List<NNLayer> layers);
	
	public List<NNLayer> getLayers();
	
	public void resetActivationCounter(boolean training);
	
	public double sgdTrain(List<Sequence> training, double learningRate, boolean shuffle);
	
	public double[][] ff(Sequence seq, ErrorFunction ef, boolean applyTraining);
	
	public void resetError();
	
	public double getError();


}