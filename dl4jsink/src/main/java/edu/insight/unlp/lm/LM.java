package edu.insight.unlp.lm;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.StringTokenizer;

import edu.insight.unlp.otdf.BasicFileTools;

/*
 * L.M. trigram based MLE estimator for P(w1|w0w-1)
 */

public class LM {

	private Set<String> vocab = new HashSet<String>();	
	private Map<String, Integer> bigramCountMap = new HashMap<String, Integer>();
	private Map<String, Integer> secondOrderMarkovCountMap = new HashMap<String, Integer>();
	public final static String PRESTART = "*-1";	
	public final static String START = "*";
	public final static String STOP = "*STOP*";
	public final static String DEPENDENTSEPARATOR = ":::";
	public final static String RESULTANTSEPARATOR = "->";
	public double TOTALBIGRAMCOUNT = 0.01;

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
				String markovBiToken = previousToken + DEPENDENTSEPARATOR + token + RESULTANTSEPARATOR + nextToken;
				String biToken = previousToken + DEPENDENTSEPARATOR + token;
				int count = 0;
				if(secondOrderMarkovCountMap.get(markovBiToken)!=null){
					count = secondOrderMarkovCountMap.get(markovBiToken);
				}
				int biGramCount = 1;
				if(bigramCountMap.get(biToken)!=null){
					biGramCount = bigramCountMap.get(biToken);
				}
				//System.out.println(markovBiToken + "\t" + count);
				double p = (double) count / (double) biGramCount;
				if(p == 0.0){
					p = 1/(double) TOTALBIGRAMCOUNT;
				}
				probCal = probCal * p;
				previousToken = token;
				token = nextToken;
			}
		}			

		return probCal;
	}

	public void learnLM(String lmDir) {		
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
						vocab.add(nextToken);
						if(!"".equals(nextToken)){
							vocab.add(nextToken);
							String markovBiToken = previousToken + DEPENDENTSEPARATOR + token + RESULTANTSEPARATOR + nextToken;
							String biToken = previousToken + DEPENDENTSEPARATOR + token;
							if(bigramCountMap.containsKey(biToken)){
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
			TOTALBIGRAMCOUNT = TOTALBIGRAMCOUNT + bigramCountMap.get(bigram);
		}
	}

	public double perplexityLM(String testDir) {
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
					double prob = getProb(line);
					logScore = logScore + Math.log(prob) / Math.log(2);			
				}
			}  	 catch (Exception e){
				e.printStackTrace();
			}
		}
		System.out.println(vocabTestData.size());
		return Math.pow(2, -logScore/vocabTestData.size());		
	}

	public static void main(String[] args) {
		LM lm = new LM();
		String trainDir = "data/lm/sampletrain";
		String testDir = "data/lm/test";
		lm.learnLM(trainDir);
		System.out.println(lm.perplexityLM(testDir));
	}

}
