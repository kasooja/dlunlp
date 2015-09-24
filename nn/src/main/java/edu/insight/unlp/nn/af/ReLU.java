package edu.insight.unlp.nn.af;

import edu.insight.unlp.nn.ActivationFunction;

public class ReLU extends ActivationFunction {

	public static double slope = 0.05;
	
	public double activation(double input) {
		if (input >= 0) {
			return input;
		}
		else {
			return input * slope;
		}
	}

	public double activationDerivative(double input) {
		if (input >= 0) {
			return 1.0;
		}
		else {
			return slope;
		}
	}

}
