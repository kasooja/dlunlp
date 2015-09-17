
package edu.insight.unlp.nn.mlp;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.NNLayer;
import edu.insight.unlp.nn.af.ReLU;
import edu.insight.unlp.nn.af.Sigmoid;
import edu.insight.unlp.nn.common.InputLayer;
import edu.insight.unlp.nn.common.SoftMaxLayer;
import edu.insight.unlp.nn.ef.SquareErrorFunction;

public class DigitClassifier {

	public static int argmax(double[] a) {
		double max = Double.MIN_VALUE;
		int maxi = -1;
		for (int i=0; i<a.length; i++) {
			if (a[i]>max) {
				max = a[i];
				maxi = i;
			}
		}
		if (maxi == -1) {
			throw new IllegalArgumentException();
		} else {
			return maxi;
		}
	}

	public static double test(NN network, double[][] inputs, double[][] targets) {
		int correct = 0;
		for (int i=0; i<inputs.length; i++) {
			int recognizedDigit = argmax(network.output(inputs[i]));
			int actualDigit = argmax(targets[i]);
			if (recognizedDigit == actualDigit) {
				correct++;
			}
		}
		return ((double)correct)/inputs.length;
	}

	public static double[][] parseFile(BufferedReader r, int numLines, int numColumns) throws IOException {
		double[][] lines = new double[numLines][];
		int i=0;
		while (r.ready()) {
			String line = r.readLine();
			String[] columns = line.split("\\s+");
			if (columns.length != numColumns) {
				System.err.println("Warning: invalid number of columns in "+r+" line "+(i+1)+" (actual "+columns.length+", expected "+numColumns+")");
			}
			lines[i] = new double[numColumns];
			for (int j=0; j<numColumns; j++) {
				lines[i][j] = Double.parseDouble(columns[j]);
			}
			i++;
		}
		if (i != numLines) {
			System.err.println("Warning: invalid number of lines (actual "+ i +", expected "+numLines+")");
		}
		return lines;
	}

	public static void main(String[] args) {
		String traindataFile = "src/test/resources/data/DigitClassifier/trainData/traindata";
		String traintargetsFile = "src/test/resources/data/DigitClassifier/trainData/traintargets";
		String testdataFile = "src/test/resources/data/DigitClassifier/testData/testdata";
		String testtargetsFile = "src/test/resources/data/DigitClassifier/testData/testtargets";
		NN nn = new MLP(new SquareErrorFunction());
		//NN nn = new MLP(new CrossEntropyErrorFunction());
		//SoftMaxLayer softmaxLayer = new SoftMaxLayer(10, nn);
		FullyConnectedLayer outputLayer = new FullyConnectedLayer(10, new Sigmoid(), nn);		
		FullyConnectedLayer hiddenLayer = new FullyConnectedLayer(40, new ReLU(), nn);		
		InputLayer inputLayer = new InputLayer(256);
		List<NNLayer> layers = new ArrayList<NNLayer>();
		layers.add(inputLayer);
		layers.add(hiddenLayer);
		layers.add(outputLayer);
		//layers.add(softmaxLayer);
		nn.setLayers(layers);
		nn.initializeNN();
		double momentum = 0.8;
		try {
			System.err.print("Reading data...");
			double[][] trainingData = parseFile(new BufferedReader(new FileReader(traindataFile)), 3000, 256);
			double[][] trainingTargets = parseFile(new BufferedReader(new FileReader(traintargetsFile)), 3000, 10);
			double[][] testData = parseFile(new BufferedReader(new FileReader(testdataFile)), 100, 256);
			double[][] testTargets = parseFile(new BufferedReader(new FileReader(testtargetsFile)), 100, 10);
			System.err.println("done.");
			int epoch = 0;
			double correctlyClassified;
			int batchSize = trainingData.length/200;
			do {
				epoch++;
				double trainingError = nn.batchgdTrain(trainingData, trainingTargets, 0.01, batchSize, true, momentum);
				int ce = ((int)(Math.exp(-trainingError)*100));
				System.out.println("epoch "+epoch+" training error: "+trainingError+" (confidence "+ce+"%)");
				correctlyClassified = test(nn, testData, testTargets);
				System.out.println((int)(correctlyClassified*100)+"% correctly classified");
			} while (correctlyClassified < 0.95);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}