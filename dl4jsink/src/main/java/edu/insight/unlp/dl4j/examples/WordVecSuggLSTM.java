package edu.insight.unlp.dl4j.examples;

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
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.BaseDatasetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**Example: Given a movie review (raw text), classify that movie review as either positive or negative based on the words it contains.
 * This is done by combining Word2Vec vectors and a recurrent neural network model. Each word in a review is vectorized
 * (using the Word2Vec model) and fed into a recurrent neural network.
 * Training data is the "Large Movie Review Dataset" from http://ai.stanford.edu/~amaas/data/sentiment/
 *
 * Process:
 * 1. Download data (movie reviews) + extract. Download + extraction is done automatically.
 * 2. Load existing Word2Vec model (for example: Google News word vectors. You will have to download this MANUALLY)
 * 3. Load each each review. Convert words to vectors + reviews to sequences of vectors
 * 4. Train network for multiple epochs. At each epoch: evaluate performance on the test set.
 *
 * NOTE / INSTRUCTIONS:
 * You will have to download the Google News word vector model manually. ~1.5GB before extraction.
 * The Google News vector model available here: https://code.google.com/p/word2vec/
 * Download the GoogleNews-vectors-negative300.bin.gz file, and extract to a suitable location
 * Then: set the WORD_VECTORS_PATH field to point to this location.
 *
 * @author Alex Black        
 */
public class WordVecSuggLSTM {          

	/** Data URL for downloading */
	//public static final String DATA_URL = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz";
	/** Location to save and extract the training/testing data */
	//public static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_w2vSentiment/");
	public static final String DATA_PATH = "/home/kartik/Dropbox/kat/ACL2016/";
	//public static final String DATA_PATH = "/src/main/resources/suggData/";
	//public static final String DATA_PATH = "/home/kartik/git/dlunlp/dl4jsink/src/main/resources/suggData";
	
	/** Location (local file system) for the Google News vectors. Set this manually. */
	//public static final String WORD_VECTORS_PATH = "/PATH/TO/YOUR/VECTORS/GoogleNews-vectors-negative300.bin";
	//public static final String WORD_VECTORS_PATH = "/home/kartik/Work/dhundo-dobara/Corpus/ML/Corpus/GoogleNews-vectors-negative300.bin";
	public static final String WORD_VECTORS_PATH_GLOVE = "/home/kartik/git/dlunlp/dl4jsink/src/main/resources/models/glove.6B.50d.txt";
	public static final String WORD_VECTORS_PATH_COMPOSES = "/home/kartik/Downloads/Mac-Downloads/John/STS/Publications/"
			+ "EN-wform.w.5.cbow.neg10.400.subsmpl.txt";	
	public static void main(String[] args) throws Exception {
		//Download and extract data
		//downloadData();

		int batchSize = 20;     //Number of examples in each minibatch
		int vectorSize = 401;   //Size of the word vectors. 300 in the Google News model
		int nEpochs = 20;        //Number of epochs (full passes of training data) to train on
		int truncateReviewsToLength = 250;  //Truncate reviews with length (# words) greater than this

		//Set up network configuration
		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
				.updater(Updater.RMSPROP)
				.regularization(true).l2(1e-5)
				.weightInit(WeightInit.XAVIER)
				.gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
				.learningRate(0.004)
				.list(2)
				.layer(0, new GravesLSTM.Builder().nIn(vectorSize).nOut(100)
						.activation("softsign").build())
				//.layer(1, new GravesLSTM.Builder().nIn(50).nOut(25)
				//		.activation("softsign").build())
				.layer(1, new RnnOutputLayer.Builder().activation("softmax")
						.lossFunction(LossFunctions.LossFunction.MCXENT).nIn(100).nOut(2).build())
				.pretrain(false).backprop(true).build();

		MultiLayerNetwork net = new MultiLayerNetwork(conf);
		net.init();
		//net.setListeners(new ScoreIterationListener(10));

		//DataSetIterators for training and testing respectively
		//Using AsyncDataSetIterator to do data loading in a separate thread; this may improve performance vs. waiting for data to load
		//WordVectors wordVectors = WordVectorSerializer.loadGoogleModel(new File(WORD_VECTORS_PATH), true, false);

		//WordVectors wordVectors = WordVectorSerializerExtended.loadTxtVectors(new File(WORD_VECTORS_PATH_GLOVE));
		WordVectors wordVectors = WordVectorSerializerExtended.loadTxtVectorsComposes(new File(WORD_VECTORS_PATH_COMPOSES));

		SuggDataIterator3 trainS = new SuggDataIterator3(DATA_PATH,wordVectors,batchSize,truncateReviewsToLength,true);
		DataSetIterator train = new AsyncDataSetIterator(trainS, 1);
		
		SuggDataIterator3 testS = new SuggDataIterator3(DATA_PATH,wordVectors,100,truncateReviewsToLength,false);
		DataSetIterator test = new AsyncDataSetIterator(testS,1);

		System.out.println("Starting training");
		for( int i=0; i<nEpochs; i++ ){
			net.fit(train);
			train.reset();
			//train.serialize();
			
			System.out.println("Epoch " + i + " complete. Starting evaluation:");
			//Run evaluation. This is on 25k reviews, so can take some time
			Evaluation evaluation = new Evaluation();
			//System.out.println("Test Count : " + testIt.testCount);
			
			while(test.hasNext()){
				DataSet t = test.next();
				INDArray features = t.getFeatureMatrix();
				INDArray lables = t.getLabels();
				INDArray inMask = t.getFeaturesMaskArray();
				INDArray outMask = t.getLabelsMaskArray();
				INDArray predicted = net.output(features,false,inMask,outMask);
				System.out.print("Evaluate: ");
				evaluation.evalTimeSeries(lables,predicted,outMask);
			}
//			System.out.println();
//			System.out.println("Test Count : " + testIt.testCount);
		
			test.reset();
			System.out.println();
			System.out.println(evaluation.stats());
		}


		System.out.println("----- Example complete -----");
	}

	//    private static void downloadData() throws Exception {
	//        //Create directory if required
	//        File directory = new File(DATA_PATH);
	//        if(!directory.exists()) directory.mkdir();
	//
	//        //Download file:
	//        String archizePath = DATA_PATH + "aclImdb_v1.tar.gz";
	//        File archiveFile = new File(archizePath);
	//
	//        if( !archiveFile.exists() ){
	//            System.out.println("Starting data download (80MB)...");
	//            FileUtils.copyURLToFile(new URL(DATA_URL), archiveFile);
	//            System.out.println("Data (.tar.gz file) downloaded to " + archiveFile.getAbsolutePath());
	//            //Extract tar.gz file to output directory
	//            extractTarGz(archizePath, DATA_PATH);
	//        } else {
	//            //Assume if archive (.tar.gz) exists, then data has already been extracted
	//            System.out.println("Data (.tar.gz file) already exists at " + archiveFile.getAbsolutePath());
	//        }
	//    }

	//	private static final int BUFFER_SIZE = 4096;
	//	private static void extractTarGz(String filePath, String outputPath) throws IOException {
	//		int fileCount = 0;
	//		int dirCount = 0;
	//		System.out.print("Extracting files");
	//		try(TarArchiveInputStream tais = new TarArchiveInputStream(
	//				new GzipCompressorInputStream( new BufferedInputStream( new FileInputStream(filePath))))){
	//			TarArchiveEntry entry;
	//
	//			/** Read the tar entries using the getNextEntry method **/
	//			while ((entry = (TarArchiveEntry) tais.getNextEntry()) != null) {
	//				//System.out.println("Extracting file: " + entry.getName());
	//
	//				//Create directories as required
	//				if (entry.isDirectory()) {
	//					new File(outputPath + entry.getName()).mkdirs();
	//					dirCount++;
	//				}else {
	//					int count;
	//					byte data[] = new byte[BUFFER_SIZE];
	//
	//					FileOutputStream fos = new FileOutputStream(outputPath + entry.getName());
	//					BufferedOutputStream dest = new BufferedOutputStream(fos,BUFFER_SIZE);
	//					while ((count = tais.read(data, 0, BUFFER_SIZE)) != -1) {
	//						dest.write(data, 0, count);
	//					}
	//					dest.close();
	//					fileCount++;
	//				}
	//				if(fileCount % 1000 == 0) System.out.print(".");
	//			}
	//		}
	//
	//		System.out.println("\n" + fileCount + " files and " + dirCount + " directories extracted to: " + outputPath);
	//	}
}