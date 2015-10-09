package edu.insight.unlp.lm;

import org.crf.function.DerivableFunction;

public class DataLMProbFunction extends DerivableFunction {

	private int sizeOfVector = 3;
	private LinearInterpolateLM lm = null;
	private String dataDir = null;

	public DataLMProbFunction(LinearInterpolateLM lm, String dataDir) {
		this.lm = lm;
		this.dataDir = dataDir;
	}

	@Override
	public double[] gradient(double[] point) {
		double[] gradient = new double[sizeOfVector];
		int i = 0;		
		lm.setLambda(point);	
		for(i=0; i<sizeOfVector; i++) {
			double inverserProbOverDir = lm.getInverserProbOverDir(dataDir, i);
			gradient[i] = inverserProbOverDir;		
		}
		return gradient;
	}

	@Override
	public double value(double[] point) {
		lm.setLambda(point);
		double probOverDir = lm.getLogProbOverDir(dataDir);
		return probOverDir;
	}

	@Override
	public int size() {
		return sizeOfVector;
	}

}
