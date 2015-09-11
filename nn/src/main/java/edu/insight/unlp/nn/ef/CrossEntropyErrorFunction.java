/**
 * 
 */
package edu.insight.unlp.nn.ef;

import edu.insight.unlp.nn.ErrorFunction;

/*
 * log loss function
 */
public class CrossEntropyErrorFunction implements ErrorFunction {

    public double[] error(double[] actual, double[] predicted) {
        double[] error = new double[actual.length+1];
        double crossEntropyError = 0.0;
        for (int i=0; i<actual.length; i++) {
            error[i] = (predicted[i]-actual[i]) * (1/((predicted[i] * (1-predicted[i]))));
            crossEntropyError -= (actual[i]) * Math.log((predicted[i])) + (1-actual[i]) * Math.log((1-predicted[i]));
        }
        error[actual.length] = crossEntropyError;
        return error;
    }
    
}
