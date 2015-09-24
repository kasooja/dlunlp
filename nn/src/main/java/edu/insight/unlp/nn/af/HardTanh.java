package edu.insight.unlp.nn.af;

import edu.insight.unlp.nn.ActivationFunction;

public class HardTanh extends ActivationFunction {

	public double activation(double input) {
		if(input>1.0){
			return 1.0;
		} else if(input<-1.0){
			return -1.0;
		} 
		return input; 
	}

	public double activationDerivative(double input) {
		return 1 - (Math.pow(activation(input), 2));
	}

}
