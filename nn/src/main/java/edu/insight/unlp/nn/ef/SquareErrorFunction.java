/**
 * 
 */
package edu.insight.unlp.nn.ef;

import java.io.Serializable;

import edu.insight.unlp.nn.ErrorFunction;


public class SquareErrorFunction implements ErrorFunction, Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	public double[] error(double[] actual, double[] predicted) {
		double[] error = new double[predicted.length+1];
		if(actual==null)
			return error;
		double squareError = 0.0;
		for (int i=0; i<predicted.length; i++) {
			error[i] = (predicted[i]-actual[i]);
			squareError += error[i] * error[i];
		}
		error[actual.length] = squareError * 0.5;
		return error;
	}

}
