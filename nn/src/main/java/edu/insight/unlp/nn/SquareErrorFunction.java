/**
 * 
 */
package edu.insight.unlp.nn;


public class SquareErrorFunction implements ErrorFunction {

	public double[] error(double[] actual, double[] predicted) {
		double[] error = new double[actual.length+1];
		double squareError = 0.0;
		for (int i=0; i<predicted.length; i++) {
			error[i] = actual[i]-predicted[i];
			squareError += error[i] * error[i];
		}
		error[actual.length] = squareError * 0.5;
		return error;
	}

}
