package edu.insight.unlp.nn;

public interface ActivationFunction {
	public double activation(double input);
	public double activationDerivative(double input);
}
