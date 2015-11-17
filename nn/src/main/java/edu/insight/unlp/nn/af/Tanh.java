package edu.insight.unlp.nn.af;

import edu.insight.unlp.nn.ActivationFunction;

public class Tanh extends ActivationFunction {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public double activation(double input) {
		if(input>20.0){
			return 1.0;
		} else if(input<-20.0){
			return -1.0;
		}
		return Math.tanh(input); //(Math.exp(input) - Math.exp(-input)) / (Math.exp(input) + Math.exp(-input)); 
	}

	public double activationDerivative(double input) {
		double coshx = Math.cosh(input);
		double denom = (Math.cosh(2*input) + 1);
		return 4 * coshx * coshx / (denom * denom); // 1 - (Math.pow(activation(input), 2));
	}
}
