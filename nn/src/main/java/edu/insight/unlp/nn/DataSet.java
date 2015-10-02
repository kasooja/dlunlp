package edu.insight.unlp.nn;

import java.util.List;
import edu.insight.unlp.nn.common.Sequence;

public abstract class DataSet {

	public int inputUnits;
	public int outputUnits;
	public ErrorFunction trainingError;
	public List<Sequence> training;
	public List<Sequence> testing;
	
	public abstract String evaluateTest(NN nn);

	public abstract void setDataSet();
	
}
