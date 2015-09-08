package edu.insight.unlp.nn;

public interface ErrorFunction {

	public double[] error(double[] actualOutput, double[] predictedOutput);
	
}