package edu.insight.unlp.nn.af;

import edu.insight.unlp.nn.ActivationFunction;

public class ReLU implements ActivationFunction {

	public double activation(double input) {
		if(input<0){
			return 0.01 * input; 
		} else {
			return input;
		}
		//return Math.log(1 + Math.exp(input));
	}

	public double activationDerivative(double input) {
		if(input<0){
			return 0.01;
		} else {
			return 1.0;
		}
		//return 1 / (1 + Math.exp(-input));
	}

}
