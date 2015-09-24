
package edu.insight.unlp.nn.rnn;

import java.util.ArrayList;
import java.util.List;

import edu.insight.unlp.nn.NNLayer;
import edu.insight.unlp.nn.RNN;
import edu.insight.unlp.nn.af.Linear;
import edu.insight.unlp.nn.af.Sigmoid;
import edu.insight.unlp.nn.common.Sequence;
import edu.insight.unlp.nn.ef.SquareErrorFunction;
import edu.insight.unlp.nn.mlp.FullyConnectedLayer;
import edu.insight.unlp.nn.utils.TemporalXOR;

public class XorRNN {

	public static double test(RNN network, List<Sequence> seqs) {
		int correct = 0;
		int totalLength = 0;
		for(Sequence seq : seqs) {
			double[][] inputSeq = seq.inputSeq;
			double[][] target = seq.target;
			double[][] output = network.output(inputSeq);
			totalLength = totalLength +  inputSeq.length;
			for(double[] out : output){
				out[0] = Math.round(out[0]);
				
			}
		//	boolean equal = true;
			for(int i=1; i<output.length; i++){
				if(target[i][0] != output[i][0]){
		//			equal = false;
				} else {
					correct++;
				}
			}
//			if(equal){
//				correct++;
//			}
			network.resetActivationCounter(false);				
		}			
		return ((double)correct)/totalLength;
	}

	public static void main(String[] args) {
		RNN nn = new RNNImpl(new SquareErrorFunction());
		FullyConnectedLayer outputLayer = new FullyConnectedLayer(1, new Sigmoid(), nn);
		FullyConnectedRNNLayer hiddenLayer = new FullyConnectedRNNLayer(10, new Sigmoid(), nn);
		FullyConnectedLayer inputLayer = new FullyConnectedLayer(1, new Linear(), nn);
		List<NNLayer> layers = new ArrayList<NNLayer>();
		layers.add(inputLayer);
		layers.add(hiddenLayer);
		layers.add(outputLayer);
		nn.setLayers(layers);
		nn.initializeNN();
		System.err.print("Reading data...");
		List<Sequence> trainSeqs = TemporalXOR.generate(1000);
		List<Sequence> testSeqs = TemporalXOR.generate(100);
		System.err.print("Done");
		int epoch = 0;
		double correctlyClassified;
		do {
			epoch++;
			double trainingError = nn.sgdTrain(trainSeqs, 0.0001, true);
			System.out.println("epoch "+epoch+" training error: " + trainingError);
			correctlyClassified = test(nn, testSeqs);
			System.out.println((int)(correctlyClassified*100)+"% correctly classified");
		} while (correctlyClassified < 0.9);
	}

}