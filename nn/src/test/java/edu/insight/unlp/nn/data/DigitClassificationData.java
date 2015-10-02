package edu.insight.unlp.nn.data;

import java.io.BufferedReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import edu.insight.unlp.nn.DataSet;
import edu.insight.unlp.nn.NN;
import edu.insight.unlp.nn.common.Sequence;
import edu.insight.unlp.nn.utils.BasicFileTools;

public class DigitClassificationData extends DataSet {

	private String traindataFile = "src/test/resources/data/DigitClassifier/trainData/traindata";
	private String traintargetsFile = "src/test/resources/data/DigitClassifier/trainData/traintargets";
	private String testdataFile = "src/test/resources/data/DigitClassifier/testData/testdata";
	private String testtargetsFile = "src/test/resources/data/DigitClassifier/testData/testtargets";
	
	private static int dataVectorSize = 256;
	private static int targetVectorSize = 10;

	public DigitClassificationData(){
		setDataSet();
	}

	@Override
	public void setDataSet() {
		this.training = getSequences(traindataFile, traintargetsFile);
		this.testing = getSequences(testdataFile, testtargetsFile);
		inputUnits = training.get(0).inputSeq[0].length;
		setDimensions();
	}
	
	private void setDimensions(){
		inputUnits = training.get(0).inputSeq[0].length;
		for(Sequence seq : training) {
			if(seq.target!=null){		
				outputUnits = seq.target[0].length;
				break;
			}
		}
	}

	private List<Sequence> getSequences(String dataFile, String targetsFile){
		List<double[][]> data = parseFile(dataFile, dataVectorSize);
		List<double[][]> targets = parseFile(targetsFile, targetVectorSize);
		List<Sequence> sequences = new ArrayList<Sequence>();
		if(data.size() == targets.size()){
			for(int i=0; i<data.size(); i++){
				Sequence seq = new Sequence(data.get(i), targets.get(i));
				sequences.add(seq);
			}		
		} else {
			System.err.println("Warning: sizes of data and targets are not equal");
		}
		return sequences;
	}

	private static List<double[][]> parseFile(String filePath, int numColumns){
		List<double[][]> data = new ArrayList<double[][]>();
		BufferedReader br = BasicFileTools.getBufferedReader(filePath);
		try {
			while (br.ready()) {
				double[][] lines = new double[1][];
				int i=0;
				String line = br.readLine();
				String[] columns = line.split("\\s+");
				if (columns.length != numColumns) {
					System.err.println("Warning: invalid number of columns in "+br+" line "+(i+1)+" (actual "+columns.length+", expected "+numColumns+")");
				}
				lines[i] = new double[numColumns];
				for (int j=0; j<numColumns; j++) {
					lines[i][j] = Double.parseDouble(columns[j]);
				}
				i++;
				data.add(lines);
			}
		} catch (NumberFormatException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}
		return data;
	}

	private static int argmax(double[] a) {
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

	@Override
	public String evaluateTest(NN nn) {
		int totalCorrect = 0;
		int totalSteps = 0;	
		StringBuilder report = new StringBuilder();
		for(Sequence seq : testing){
			int recognizedDigit = argmax(nn.output(seq.inputSeq)[0]);
			int actualDigit = argmax(seq.target[0]);
			if (recognizedDigit == actualDigit) {
				totalCorrect++;
			}
			totalSteps = totalSteps + seq.inputSeq.length;
			nn.resetActivationCounter(false);
		}
		double correctlyClassified = ((double)totalCorrect/(double)totalSteps) * 100;
		report.append((int)correctlyClassified + "% correctly classified");
		return report.toString();
	}

}
