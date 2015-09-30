package edu.insight.unlp.nn.common;

import java.util.Random;
import java.util.stream.IntStream;

import org.apache.commons.math.random.RandomDataImpl;

public class WeightInitializer {

	private static Random rng = new Random();
	
	/**
	 * Sets the weights in the whole matrix uniformly between -eInit and eInit
	 * (eInit is the standard deviation) with zero mean.
	 */
	private static void setWeightsUniformly(RandomDataImpl rnd, double eInit, double[] weights) {
		for (int i = 0; i < weights.length; i++) {		
			weights[i] = rnd.nextUniform(-eInit, eInit);
		}
	}

	private static RandomDataImpl seedRandomGenerator() {
		RandomDataImpl rnd = new RandomDataImpl();
		rnd.reSeed(System.currentTimeMillis());
		rnd.reSeedSecure(System.currentTimeMillis());
		return rnd;
	}

	public static void randomInitializeKarapathyCode(WeightMatrix weightMatrix, double initParamsStdDev, Double biasInitialVal){
		for (int i = 0; i < weightMatrix.weights.length; i++) {
			weightMatrix.weights[i] = rng.nextGaussian() * initParamsStdDev;
		}
		if(biasInitialVal!=null) {
			initializeSeperateBias(weightMatrix, biasInitialVal);
		}
	}

	private static void initializeSeperateBias(WeightMatrix weightMatrix, Double biasInitialVal){
		int counter = 1;
		for(int i=0; i<weightMatrix.weights.length; i=counter++*weightMatrix.biasMultiplier){
			weightMatrix.weights[i] = biasInitialVal;
		}
	}

	public static void randomInitializeLeCun(WeightMatrix weightMatrix, Double biasInitialVal){
		double eInit = Math.sqrt(6) / Math.sqrt(weightMatrix.weights.length - 1);
		setWeightsUniformly(seedRandomGenerator(), eInit, weightMatrix.weights);
		if(biasInitialVal!=null) {
			initializeSeperateBias(weightMatrix, biasInitialVal);
		}
	}

	public static void constantInitialize(WeightMatrix weightMatrix, Double val, Double biasInitialVal){
		IntStream.range(0, weightMatrix.weights.length).forEach(i -> weightMatrix.weights[i] = val);
		if(biasInitialVal!=null) {
			initializeSeperateBias(weightMatrix, biasInitialVal);
		}
	}

	public static void randomInitialize(WeightMatrix weightMatrix, Double biasInitialVal){
		IntStream.range(0, weightMatrix.weights.length).forEach(i -> weightMatrix.weights[i] = (Math.random() * 2 - 1));
		if(biasInitialVal!=null) {
			initializeSeperateBias(weightMatrix, biasInitialVal);
		}
	}

}
