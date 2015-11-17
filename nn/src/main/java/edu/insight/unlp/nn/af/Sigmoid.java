package edu.insight.unlp.nn.af;

import edu.insight.unlp.nn.ActivationFunction;

public class Sigmoid extends ActivationFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public double activation(double input) {
		return 1.0 / (1 + Math.exp(-input));
	}

	public double activationDerivative(double input) {
		return activation(input) * (1-activation(input));
	}

}
