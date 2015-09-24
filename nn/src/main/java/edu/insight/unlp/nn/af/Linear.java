package edu.insight.unlp.nn.af;

import edu.insight.unlp.nn.ActivationFunction;

public class Linear extends ActivationFunction {

	@Override
	public double activation(double input) {
		return input;
	}

	@Override
	public double activationDerivative(double input) {
		return 1;
	}

}

