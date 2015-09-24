package edu.insight.unlp.nn;

import java.util.stream.IntStream;

public abstract class ActivationFunction {
	
	public double[] activation(double[] input){
		double[] activations = new double[input.length];
		IntStream.range(0, input.length).forEach(i -> activations[i] = activation(input[i]));
		return activations;
	}
	public double[] activationDerivative(double[] input){
		double[] derivatives = new double[input.length];
		IntStream.range(0, input.length).forEach(i -> derivatives[i] = activationDerivative(input[i]));
		return derivatives;
	}
	
	public abstract double activation(double input);
	public abstract double activationDerivative(double input);
	
}
