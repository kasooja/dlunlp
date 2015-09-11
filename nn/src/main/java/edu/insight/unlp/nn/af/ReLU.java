package edu.insight.unlp.nn.af;

import edu.insight.unlp.nn.ActivationFunction;

public class ReLU implements ActivationFunction {

	public double activation(double input) {
		return Math.log(1 + Math.exp(input));
	}

	public double activationDerivative(double input) {
		return 1 / (1 + Math.exp(-input));
	}

}
