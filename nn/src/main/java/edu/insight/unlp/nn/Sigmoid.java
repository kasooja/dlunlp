package edu.insight.unlp.nn;

public class Sigmoid implements ActivationFunction {

	public double activation(double input) {
		return 1.0 / (1 + Math.exp(-input));
	}

	public double activationDerivative(double input) {
		return activation(input) * (1-activation(input));
	}

}
