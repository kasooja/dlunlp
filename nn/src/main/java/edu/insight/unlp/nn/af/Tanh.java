package edu.insight.unlp.nn.af;

import edu.insight.unlp.nn.ActivationFunction;

public class Tanh implements ActivationFunction {

	public double activation(double input) {
		return (Math.exp(input) - Math.exp(-input)) / (Math.exp(input) + Math.exp(-input)); 
	}

	public double activationDerivative(double input) {
		return 1 - (Math.pow(activation(input), 2));
	}

}
