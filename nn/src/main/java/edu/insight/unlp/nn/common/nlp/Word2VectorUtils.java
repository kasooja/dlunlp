package edu.insight.unlp.nn.common.nlp;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

public class Word2VectorUtils {
	
	public static double getSim(double[] x, double[] y) {
		INDArray array1 = Nd4j.create(x);
		INDArray array2 = Nd4j.create(y);
		INDArray vector1 = Transforms.unitVec(array1);
		INDArray vector2 = Transforms.unitVec(array2);
		if(vector1 == null || vector2 == null)
			return -1;
		return  Nd4j.getBlasWrapper().dot(vector1, vector2);
	}
	
}
