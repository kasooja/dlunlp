
package edu.insight.unlp.nn.examples;

import java.util.ArrayList;
import java.util.List;

import edu.insight.unlp.nn.DataSet;
import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.NNImpl;
import edu.insight.unlp.nn.NNLayer;
import edu.insight.unlp.nn.af.Linear;
import edu.insight.unlp.nn.af.Sigmoid;
import edu.insight.unlp.nn.common.nlp.GloveVectors;
import edu.insight.unlp.nn.common.nlp.Word2Vector;
import edu.insight.unlp.nn.data.GRCTCModalityClassificationData;
import edu.insight.unlp.nn.data.GRCTCProvisionClassificationData;
import edu.insight.unlp.nn.data.SuggestionClassificationData;
import edu.insight.unlp.nn.ef.SquareErrorFunction;
import edu.insight.unlp.nn.layers.FullyConnectedFFLayer;
import edu.insight.unlp.nn.layers.FullyConnectedLSTMLayer;
import edu.insight.unlp.nn.layers.FullyConnectedRNNLayer;

public class RNNExample {

	public static void main(String[] args) {
	//	System.err.print("Reading data...");
		Word2Vector word2vec = new GloveVectors();
		DataSet dataset = new SuggestionClassificationData(word2vec);
		//System.err.println("Done");
		NN nn = new NNImpl(new SquareErrorFunction());
		NNLayer outputLayer = new FullyConnectedFFLayer(dataset.outputUnits, new Sigmoid(), nn);
		//NNLayer hiddenLayer = new LS(10, nn);
//		NNLayer hiddenLayer2 = new FullyConnectedRNNLayer(14, new Sigmoid(), nn);
//		NNLayer hiddenLayer = new FullyConnectedRNNLayer(35, new Sigmoid(), nn);
		NNLayer hiddenLayer2 = new FullyConnectedLSTMLayer(20, nn);
		NNLayer hiddenLayer = new FullyConnectedLSTMLayer(40, nn);
		//NNLayer hiddenLayer = new FullyConnectedFFLayer(10, new Sigmoid(), nn);
		NNLayer inputLayer = new FullyConnectedFFLayer(dataset.inputUnits, new Linear(), nn);
		List<NNLayer> layers = new ArrayList<NNLayer>();
		layers.add(inputLayer);
		layers.add(hiddenLayer);
		layers.add(hiddenLayer2);
		layers.add(outputLayer);
		nn.setLayers(layers);
		nn.initializeNN();
		
		int epoch = 0;
		int maxEpochs = 140;
		int evaluateEveryNthEpoch = 2;
		while(epoch<maxEpochs) {
			epoch++;
			double trainingError = nn.sgdTrain(dataset.training, 0.001, true);
			System.out.println("epoch["+epoch+"/" + maxEpochs + "] train loss = " + trainingError);
			if(epoch%evaluateEveryNthEpoch==0)
				System.out.println(dataset.evaluateTest(nn));
		}
	}

}