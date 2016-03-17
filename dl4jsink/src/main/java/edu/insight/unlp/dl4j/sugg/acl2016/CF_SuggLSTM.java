package edu.insight.unlp.dl4j.sugg.acl2016;

import java.io.File;

import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
public class CF_SuggLSTM {          


	public static final String DATA_PATH = "/home/kartik/git/dlunlp/dl4jsink/src/main/resources/suggData/electronics.csv";

	public static final String WORD_VECTORS_PATH_GOOGLE = "/home/kartik/Work/dhundo-dobara/Corpus/ML/Corpus/GoogleNews-vectors-negative300.bin";
	public static final String WORD_VECTORS_PATH_GLOVE = "/home/kartik/git/dlunlp/dl4jsink/src/main/resources/models/glove.6B.50d.txt";
	public static final String WORD_VECTORS_PATH_COMPOSES = "/home/kartik/Downloads/Mac-Downloads/John/STS/Publications/"
			+ "EN-wform.w.5.cbow.neg10.400.subsmpl.txt";	

	public static void main(String[] args) throws Exception {
		int trainBatchSize = 10;     //Number of examples in each minibatch
		int testBatchSize = 500;
		int vectorSize = 51;   //Size of the word vectors size, 400 in Composes, 50 in small Glove, 300 in Google
		int nEpochs = 5;        //Number of epochs (full passes of training data) to train on
		int truncateReviewsToLength = 50;  //Truncate reviews with length (# words) greater than this

		//Set up network configuration
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
				.updater(Updater.RMSPROP)
				.regularization(true).l2(1e-5)
				.weightInit(WeightInit.XAVIER)
				.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
				.learningRate(0.004)
				.list(2)
				.layer(0, new GravesLSTM.Builder().nIn(vectorSize).nOut(25)
						.activation("softsign").build())
				//.layer(1, new GravesLSTM.Builder().nIn(50).nOut(25)
				//		.activation("softsign").build())
				.layer(1, new RnnOutputLayer.Builder().activation("softmax")
						.lossFunction(LossFunctions.LossFunction.MCXENT).nIn(25).nOut(2).build())
				.pretrain(false).backprop(true).build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		//net.setListeners(new ScoreIterationListener(10));

		//WordVectors wordVectors = WordVectorSerializer.loadGoogleModel(new File(WORD_VECTORS_PATH), true, false);
		WordVectors wordVectors = Dl4j_ExtendedWVSerializer.loadTxtVectors(new File(WORD_VECTORS_PATH_GLOVE));
		//WordVectors wordVectors = WordVectorSerializerExtended.loadTxtVectorsComposes(new File(WORD_VECTORS_PATH_COMPOSES));

		int noOfFolds = 4;

		CF_SuggDataProducer cfiterator = new CF_SuggDataProducer(noOfFolds, DATA_PATH, wordVectors, trainBatchSize, testBatchSize, truncateReviewsToLength, WVPOS_SuggDataIterator.class);

		double[] foldPs = new double[noOfFolds];
		double[] foldRs = new double[noOfFolds];
		double[] foldF1s = new double[noOfFolds];

		for(int fold=0; fold<noOfFolds; fold++){
			Pair<DataSetIterator, DataSetIterator> suggData = cfiterator.nextPair();
			DataSetIterator trainIt = suggData.getFirst();
			DataSetIterator testIt = suggData.getSecond();

			DataSetIterator train = new AsyncDataSetIterator(trainIt, 1);
			DataSetIterator test = new AsyncDataSetIterator(testIt,1);

			System.out.println("Starting training");
			System.out.println("Fold No: " + fold);

			for( int i=0; i<nEpochs; i++ ){
				net.fit(train);
				train.reset();
				System.out.println("Epoch " + i + " complete.");
			}

			System.out.println("Starting evaluation:");
			Evaluation evaluation = new Evaluation();
			int totalExamples = 0;
			int batchNo = 0;
			while(test.hasNext()){
				DataSet t = test.next();
				int numExamples = t.numExamples();
				INDArray features = t.getFeatureMatrix();
				INDArray labels = t.getLabels();
				INDArray inMask = t.getFeaturesMaskArray();
				INDArray outMask = t.getLabelsMaskArray();
				INDArray predicted = net.output(features, false, inMask, outMask);
				System.out.print(batchNo++ + " ");
				evaluation.evalTimeSeries(labels, predicted, outMask);
				totalExamples = totalExamples + numExamples;
			}
			test.reset();

			System.out.println();
			System.out.println("Fold " + fold + " Results");
			System.out.println("No. of Examples in this Fold: " + totalExamples);
			System.out.println(evaluation.stats());
			printPRF(evaluation, 0, 1);

			foldPs[fold] = evaluation.precision(0);
			foldRs[fold] = evaluation.recall(0);
			foldF1s[fold] = evaluation.f1(0);
			System.out.println();
		}
		System.out.println("Average statistics ");
		double sumP = 0.0;
		double sumR = 0.0;
		double sumF1 = 0.0;

		for(double p : foldPs){
			sumP = sumP + p;
		}
		for(double r : foldRs){
			sumR = sumR + r;
		}
		for(double f1 : foldF1s){
			sumF1 = sumF1 + f1;
		}
		
		double avgP = sumP/foldPs.length;
		double avgR = sumR/foldRs.length;
		double avgF1 = sumF1/foldF1s.length;

		System.out.print("Avg P: " + avgP + " ");
		System.out.print("Avg R: " + avgR + " ");
		System.out.println("Avg F1: " + avgF1 + " ");
		System.out.println("Computed Avg F1: " + (2.0 * avgP * avgR)/(avgP + avgR) + " ");

		System.out.println("----- Evaluation complete -----");
	}

	public static void printPRF(Evaluation evaluation, int... classLabels){
		for(int classLabel : classLabels){
			System.out.println("Class " + classLabel +  ": ");
			System.out.print("P: " + evaluation.precision(classLabel) + " ");
			System.out.print("R: " + evaluation.recall(classLabel) + " ");
			System.out.println("F1: " + evaluation.f1(classLabel) + " ");
		}
	}

}