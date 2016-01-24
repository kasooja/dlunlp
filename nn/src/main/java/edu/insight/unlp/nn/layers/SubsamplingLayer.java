package edu.insight.unlp.nn.layers;


import edu.insight.unlp.nn.NNLayer;
import edu.insight.unlp.nn.common.WeightMatrix;
import it.unimi.dsi.fastutil.ints.Int2ObjectMap;

public class SubsamplingLayer extends NNLayer {

	@Override
	public double[] errorGradient(double[] input) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public double[] computeSignals(double[] input, WeightMatrix weights,
			Int2ObjectMap<double[]> activations) {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void initializeLayer(int previousLayerUnits) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public double[] errorGradient(double[] eg, double[] input, double[] na) {
		// TODO Auto-generated method stub
		return null;
	}

}
