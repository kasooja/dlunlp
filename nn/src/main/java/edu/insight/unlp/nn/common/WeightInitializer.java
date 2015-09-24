package edu.insight.unlp.nn.common;

import java.util.Random;
import java.util.stream.IntStream;

import org.apache.commons.math.random.RandomDataImpl;

public class WeightInitializer {

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

	public static void randomInitialize2(double[] weights, double initParamsStdDev){
		Random rng = new Random();
		for (int i = 0; i < weights.length; i++) {
			weights[i] = rng.nextGaussian() * initParamsStdDev;
		}
	}

	public static void randomInitializeLeCun(double[] weights){
		double eInit = Math.sqrt(6) / Math.sqrt(weights.length - 1);
		setWeightsUniformly(seedRandomGenerator(), eInit, weights);
	}

	public static void constantInitialize(double[] weights, double val){
		IntStream.range(0, weights.length).forEach(i -> weights[i] = val);
	}

	public static void randomInitialize(double[] weights){
		IntStream.range(0, weights.length).forEach(i -> weights[i] = (Math.random() * 2 - 1));
	}

}
