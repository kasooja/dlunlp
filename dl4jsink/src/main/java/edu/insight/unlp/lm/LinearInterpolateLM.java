package edu.insight.unlp.lm;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;

import org.crf.function.optimization.GradientDescentOptimizer;
import org.crf.function.optimization.NegatedFunction;

import edu.insight.unlp.otdf.BasicFileTools;


/*
 * Linear Interpolation method for L.M. (max trigram) based MLE estimator for P(w1|w0w-1)
 */

public class LinearInterpolateLM {

	private Set<String> vocab = new HashSet<String>();
	private Map<String, Integer> unigramCountMap = new HashMap<String, Integer>();	
	private Map<String, Integer> bigramCountMap = new HashMap<String, Integer>();	
	private Map<String, Integer> firstOrderMarkovCountMap = new HashMap<String, Integer>();
	private Map<String, Integer> secondOrderMarkovCountMap = new HashMap<String, Integer>();
	public final static String PRESTART = "*-1";	
	public final static String START = "*";
	public final static String STOP = "*STOP*";
	public final static String DEPENDENTSEPARATOR = ":::";
	public final static String RESULTANTSEPARATOR = "->";
	public double totalBigramCount = 0.0;
	public double totalUnigramCount = 0.0;
	public double lambda[] = new double[3];
	private double tempTestVocabSize = 0.0;

	public void setLambda(double[] lambda){
		this.lambda = lambda;
	}

	public double[] getLambda(){
		return lambda;
	}

	public double getLogProb(String test){
		StringTokenizer tokenizer = new StringTokenizer(test, ". ' ,", true);
		String previousToken = START;				
		String token = START;		
		previousToken = START;				
		token = START;
		double probCal = 0.0;
		while(tokenizer.hasMoreTokens()) {			
			String nextToken = tokenizer.nextToken().trim().toLowerCase();			
			if(!"".equals(nextToken)){
				String markovUniToken = token + RESULTANTSEPARATOR + nextToken;
				String markovBiToken = previousToken + DEPENDENTSEPARATOR + token + RESULTANTSEPARATOR + nextToken;

				String biToken = previousToken + DEPENDENTSEPARATOR + token;			

				int triCount = 0;
				if(secondOrderMarkovCountMap.get(markovBiToken)!=null){
					triCount = secondOrderMarkovCountMap.get(markovBiToken);
				}

				int biGramCount = 1;
				if(bigramCountMap.get(biToken)!=null){
					biGramCount = bigramCountMap.get(biToken);
				}				

				int biCount = 0;
				if(firstOrderMarkovCountMap.get(markovUniToken)!=null){
					biCount = firstOrderMarkovCountMap.get(markovUniToken);
				}

				int uniGramCount = 1;
				if(unigramCountMap.get(token)!=null){
					uniGramCount = unigramCountMap.get(token);
				}
				int uCount = 1;
				if(unigramCountMap.get(nextToken)!=null){
					uCount = unigramCountMap.get(nextToken);
				}
				double pTrigram = (double) triCount / (double) biGramCount;

				double pBigram = (double) biCount / (double) uniGramCount;			

				double pUnigram = (double) uCount / (double) totalUnigramCount;
				double lo = lambda[0]*pTrigram + lambda[1]*pBigram + lambda[2]*pUnigram;
				double p = 1 / (1+Math.pow(Math.E, -lo)); 

				if(p == 0.0){
					p = 1/(double) (2 * totalBigramCount + 2 * totalUnigramCount);
				}

				probCal = probCal +  Math.log(p);//Math.log(2);
				previousToken = token;
				token = nextToken;
			}
		}			
		return probCal;

	}

	public double getProb(String test){
		StringTokenizer tokenizer = new StringTokenizer(test, ". ' ,", true);
		String previousToken = START;				
		String token = START;		
		previousToken = START;				
		token = START;
		double probCal = 1.0;
		while(tokenizer.hasMoreTokens()) {			
			String nextToken = tokenizer.nextToken().trim().toLowerCase();			
			if(!"".equals(nextToken)){
				String markovUniToken = token + RESULTANTSEPARATOR + nextToken;
				String markovBiToken = previousToken + DEPENDENTSEPARATOR + token + RESULTANTSEPARATOR + nextToken;

				String biToken = previousToken + DEPENDENTSEPARATOR + token;			

				int triCount = 0;
				if(secondOrderMarkovCountMap.get(markovBiToken)!=null){
					triCount = secondOrderMarkovCountMap.get(markovBiToken);
				}

				int biGramCount = 1;
				if(bigramCountMap.get(biToken)!=null){
					biGramCount = bigramCountMap.get(biToken);
				}				

				int biCount = 0;
				if(firstOrderMarkovCountMap.get(markovUniToken)!=null){
					biCount = firstOrderMarkovCountMap.get(markovUniToken);
				}

				int uniGramCount = 1;
				if(unigramCountMap.get(token)!=null){
					uniGramCount = unigramCountMap.get(token);
				}
				int uCount = 1;
				if(unigramCountMap.get(nextToken)!=null){
					uCount = unigramCountMap.get(nextToken);
				}
				double pTrigram = (double) triCount / (double) biGramCount;

				double pBigram = (double) biCount / (double) uniGramCount;			

				double pUnigram = (double) uCount / (double) totalUnigramCount;

				double p = lambda[0]*pTrigram + lambda[1]*pBigram + lambda[2]*pUnigram; 

				if(p == 0.0){
					p = 1/(double) (2 * totalBigramCount + 2 * totalUnigramCount);
				}

				probCal = probCal * p;
				previousToken = token;
				token = nextToken;
			}
		}			
		return probCal;
	}


	public double getInverseProb(String test, int whichLambda){
		StringTokenizer tokenizer = new StringTokenizer(test, ". ' ,", true);
		String previousToken = START;				
		String token = START;		
		previousToken = START;				
		token = START;
		double probCal = 0.0;
		while(tokenizer.hasMoreTokens()) {			
			String nextToken = tokenizer.nextToken().trim().toLowerCase();			
			if(!"".equals(nextToken)){
				String markovUniToken = token + RESULTANTSEPARATOR + nextToken;
				String markovBiToken = previousToken + DEPENDENTSEPARATOR + token + RESULTANTSEPARATOR + nextToken;

				String biToken = previousToken + DEPENDENTSEPARATOR + token;			

				int triCount = 0;
				if(secondOrderMarkovCountMap.get(markovBiToken)!=null){
					triCount = secondOrderMarkovCountMap.get(markovBiToken);
				}

				int biGramCount = 1;
				if(bigramCountMap.get(biToken)!=null){
					biGramCount = bigramCountMap.get(biToken);
				}				

				int biCount = 0;
				if(firstOrderMarkovCountMap.get(markovUniToken)!=null){
					biCount = firstOrderMarkovCountMap.get(markovUniToken);
				}

				int uniGramCount = 1;
				if(unigramCountMap.get(token)!=null){
					uniGramCount = unigramCountMap.get(token);
				}
				int uCount = 1;
				if(unigramCountMap.get(nextToken)!=null){
					uCount = unigramCountMap.get(nextToken);
				}
				double pTrigram = (double) triCount / (double) biGramCount;

				double pBigram = (double) biCount / (double) uniGramCount;			

				double pUnigram = (double) uCount / (double) totalUnigramCount;

				//double p = lambda[0]*pTrigram + lambda[1]*pBigram + lambda[2]*pUnigram; 

				double lo = lambda[0]*pTrigram + lambda[1]*pBigram + lambda[2]*pUnigram;
				if(lo>100.0){
					lo = 100.0;
				}

				double z = 1 / (1+Math.pow(Math.E, -lo));
							double den = Math.pow((1+Math.pow(Math.E, lo)), 2);
				if(den==0.0){
					den = 0.0001;
				}
				double p =  z * (Math.pow(Math.E, lo) / den);


				if(p == 0.0){
					p = 1/(double) (2 * totalBigramCount + 2 * totalUnigramCount);
				}				

				double n = 0.0;
				//probCal = probCal + 1/p;

				if(whichLambda == 0){
					n = p * pTrigram;

				}

				if(whichLambda == 1){
					n = p * pBigram;

				}

				if(whichLambda == 2){
					n = p * pUnigram;

				}
				probCal = probCal + n;

				previousToken = token;
				token = nextToken;
			}
		}			
		return probCal;

	}
	public void learnLM(String lmDir, String validateDir) {		
		File[] files = (new File(lmDir)).listFiles();
		for(File file : files){
			BufferedReader br = BasicFileTools.getBufferedReader(file);
			String line = null;			
			try {
				while((line = br.readLine()) != null) {					
					StringTokenizer tokenizer = new StringTokenizer(line, ". ' ,", true);
					String previousToken = START;				
					String token = START;					
					while(tokenizer.hasMoreTokens()){						
						String nextToken = tokenizer.nextToken().trim().toLowerCase();						
						if(!"".equals(nextToken)){
							vocab.add(nextToken);							

							String markovUniToken = token + RESULTANTSEPARATOR + nextToken;						
							if(unigramCountMap.get(token) != null){
								unigramCountMap.put(token, unigramCountMap.get(token) + 1);
							} else {
								unigramCountMap.put(token, 1);
							}						
							if(firstOrderMarkovCountMap.get(markovUniToken)!=null){
								firstOrderMarkovCountMap.put(markovUniToken, firstOrderMarkovCountMap.get(markovUniToken) + 1);
							} else {
								firstOrderMarkovCountMap.put(markovUniToken, 1);
							}

							String markovBiToken = previousToken + DEPENDENTSEPARATOR + token + RESULTANTSEPARATOR + nextToken;
							String biToken = previousToken + DEPENDENTSEPARATOR + token;
							if(bigramCountMap.get(biToken) != null){
								bigramCountMap.put(biToken, bigramCountMap.get(biToken) + 1);
							} else {
								bigramCountMap.put(biToken, 1);
							}						
							if(secondOrderMarkovCountMap.get(markovBiToken) != null){
								secondOrderMarkovCountMap.put(markovBiToken, secondOrderMarkovCountMap.get(markovBiToken) + 1);
							} else {
								secondOrderMarkovCountMap.put(markovBiToken, 1);
							}

							previousToken = token;
							token = nextToken;		
						}
					}
				}
			} catch (IOException e) {
				e.printStackTrace();
			}

		}
		for(String bigram : bigramCountMap.keySet()){
			totalBigramCount = totalBigramCount + bigramCountMap.get(bigram);
		}
		for(String unigram : unigramCountMap.keySet()){
			totalUnigramCount = totalUnigramCount + unigramCountMap.get(unigram);
		}	




		DataLMProbFunction lmProbFunction = new DataLMProbFunction(this, validateDir);


		NegatedFunction negFunc = NegatedFunction.fromDerivableFunction(lmProbFunction);


		GradientDescentOptimizer optimizer = new GradientDescentOptimizer(negFunc);


		optimizer.find();


		double[] point = optimizer.getPoint();
		lambda = point;
		System.out.println();
		for(double d : lambda){
			System.out.println(d);
		}
	}

	public double perplexityLM(String testDir) {
		return Math.pow(2, -getLogProbOverDir(testDir)/tempTestVocabSize);	
	}

	public double getLogProbOverDir(String testDir) {
		Set<String> vocabTestData = new HashSet<String>();
		vocabTestData.add(START);
		vocabTestData.add(STOP);
		double logScore = 0.0;
		File[] files = (new File(testDir)).listFiles();
		for(File file : files){
			BufferedReader br = BasicFileTools.getBufferedReader(file);
			String line = null;			
			try {
				while((line = br.readLine()) != null) {
					StringTokenizer tokenizer = new StringTokenizer(line, ". ' ,", true);					
					while(tokenizer.hasMoreTokens()){						
						String nextToken = tokenizer.nextToken().trim().toLowerCase();
						vocabTestData.add(nextToken);
					}
					double prob = getLogProb(line);
					logScore = logScore + prob;// / Math.log(2);			
				}
			}  	 catch (Exception e){
				e.printStackTrace();
			}
		}
		//	System.out.println(vocabTestData.size());
		tempTestVocabSize = vocabTestData.size();
		return logScore;		
	}

	public double getInverserProbOverDir(String testDir, int whichLambda) {		
		Set<String> vocabTestData = new HashSet<String>();
		vocabTestData.add(START);
		vocabTestData.add(STOP);
		double inverseProbSummation = 0.0;
		File[] files = (new File(testDir)).listFiles();
		for(File file : files){
			BufferedReader br = BasicFileTools.getBufferedReader(file);
			String line = null;			
			try {
				while((line = br.readLine()) != null) {
					StringTokenizer tokenizer = new StringTokenizer(line, ". ' ,", true);					
					while(tokenizer.hasMoreTokens()){						
						String nextToken = tokenizer.nextToken().trim().toLowerCase();
						vocabTestData.add(nextToken);
					}
					double prob = getInverseProb(line, whichLambda);
					inverseProbSummation = inverseProbSummation + prob;			
				}
			}  	 catch (Exception e){
				e.printStackTrace();
			}
		}
		//	System.out.println(vocabTestData.size());
		tempTestVocabSize = vocabTestData.size();
		return inverseProbSummation;		
	}


	public static void main(String[] args) {
		LinearInterpolateLM lm = new LinearInterpolateLM();	
		String trainDir = "data/lm/train";
		String testDir = "data/lm/test";
		lm.learnLM(trainDir, testDir);
		//System.out.println(lm.perplexityLM(testDir));
	}

}
