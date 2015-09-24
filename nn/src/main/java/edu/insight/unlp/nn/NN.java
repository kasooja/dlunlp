
package edu.insight.unlp.nn;

import java.util.List;


public interface NN {
	
	public void update(double learningRate);

	public int numOutputUnits();
	
	public void initializeNN();
	
	public void setLayers(List<NNLayer> layers);
	
	public List<NNLayer> getLayers();
	
	public void resetActivationCounter(boolean training);

}