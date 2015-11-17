package edu.insight.unlp.nn.common;

import java.io.Serializable;

public class WeightMatrix implements Serializable {
	
	
	private static final long serialVersionUID = 1L;
	//flattened matrix	
	public double[] weights; //keeps the weights of the connections from the previous layer, in lstm, cellStateInput weights
	public double[] deltas;
	public double[] stepCache; // stepCache, need to explore more, it is for per parameter RMS weight update
	
	public int biasMultiplier;

}
