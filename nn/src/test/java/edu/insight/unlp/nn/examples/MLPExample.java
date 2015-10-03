package edu.insight.unlp.nn.examples;

import java.util.ArrayList;
import java.util.List;

import edu.insight.unlp.nn.DataSet;
import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.NNImpl;
import edu.insight.unlp.nn.NNLayer;
import edu.insight.unlp.nn.af.Linear;
import edu.insight.unlp.nn.af.Sigmoid;
import edu.insight.unlp.nn.data.DigitClassificationData;
import edu.insight.unlp.nn.ef.SquareErrorFunction;
import edu.insight.unlp.nn.layers.FullyConnectedFFLayer;

public class MLPExample {

	public static void main(String[] args) {
		System.err.println("Reading data...");
		DataSet dataset = new DigitClassificationData();
		System.err.println("done.");
		
		NN nn = new NNImpl(new SquareErrorFunction());
		NNLayer outputLayer = new FullyConnectedFFLayer(dataset.outputUnits, new Sigmoid(), nn);		
		NNLayer hiddenLayer = new FullyConnectedFFLayer(10, new Sigmoid(), nn);		
		NNLayer inputLayer = new FullyConnectedFFLayer(dataset.inputUnits, new Linear(), nn);
		List<NNLayer> layers = new ArrayList<NNLayer>();
		layers.add(inputLayer);
		layers.add(hiddenLayer);
		layers.add(outputLayer);
		nn.setLayers(layers);
		nn.initializeNN();
		
		int epoch = 0;
		int maxEpochs = 50;
		int evaluateEveryNthEpoch = 2;
		while(epoch<maxEpochs){
			epoch++;
			double trainingError = nn.sgdTrain(dataset.training, 0.001, true);
			System.out.println("epoch["+epoch+"/" + maxEpochs + "] train loss = " + trainingError);
			if(epoch%evaluateEveryNthEpoch==0)
				System.out.println(dataset.evaluateTest(nn));
		}
	}

}