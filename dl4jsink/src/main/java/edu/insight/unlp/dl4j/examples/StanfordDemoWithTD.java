package edu.insight.unlp.dl4j.examples;


import java.io.BufferedReader;

//performs..
//author snegi

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.insight.unlp.otdf.BasicFileTools;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.Label;
import edu.stanford.nlp.ling.Sentence;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.trees.GrammaticalStructureFactory;
import edu.stanford.nlp.trees.PennTreebankLanguagePack;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.trees.TreeCoreAnnotations.TreeAnnotation;
import edu.stanford.nlp.trees.TreebankLanguagePack;
import edu.stanford.nlp.trees.TypedDependency;
import edu.stanford.nlp.util.CoreMap;

public class StanfordDemoWithTD {


	private static StanfordCoreNLP pipeline = new StanfordCoreNLP();
	private static String patt = ".*\\((\\w+)-\\d*,\\s*(\\w+)-\\d*\\)";
	private static String patt1 = "(.+)\\((.+)-\\d+\\s*,\\s*(.+)-\\d+\\)";
	private static Pattern pattern;
	private static Pattern pattern1;
	public static String stantagsFile = "/home/kartik/git/ytip/classifier/src/main/resources/StanTags";
	public static List<String> stanTags = new ArrayList<String>();
	public static String tagsRegex = "\\d+\\.\\s+([A-Z$]+)";
	public static Pattern tagsPatt = Pattern.compile(tagsRegex);
	

	// creates a StanfordCoreNLP object, with POS tagging, lemmatization, NER, parsing, and coreference resolution 
	private static	Properties props = new Properties();
	static {
		//pattern = Pattern.compile(patt);
		//pattern1 = Pattern.compile(patt1);
		props.put("annotators", "tokenize, ssplit, parse"); //, lemma
		setTags();
	}	

	private static void setTags(){
		BufferedReader br = BasicFileTools.getBufferedReader(stantagsFile);
		String line = null;
		try {
			while((line = br.readLine())!=null){
				Matcher matcher = tagsPatt.matcher(line);
				if(matcher.find()){
					String group = matcher.group(1).trim();
					stanTags.add(group);
				}
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static StanfordCoreNLP sentPipeline = new StanfordCoreNLP(props);

	//returns Adjective Phrase, by taking a Parse Tree as input

	public static String lemmatizeWord(String text)
	{
		List<String> lemmas = new LinkedList<String>();
		Annotation document = new Annotation(text);
		// run all Annotators on this text
		pipeline.annotate(document);
		List<CoreMap> sentences = document.get(SentencesAnnotation.class);
		for(CoreMap sentence: sentences) {
			// Iterate over all tokens in a sentence
			for (CoreLabel token: sentence.get(TokensAnnotation.class)) {
				// Retrieve and add the lemma for each word into the
				// list of lemmas
				lemmas.add(token.get(LemmaAnnotation.class));
			}
		}

		String lemma = new String(lemmas.get(0));    
		return lemma;
	}

	public static String getAdjP(Tree node){
		StringBuffer buffer = new StringBuffer();
		if(!node.isLeaf())   //if it is an intermediate node of parse tree, since ADJP cannot be the leaf
		{
			Label label = node.label();
			if(label.toString().equalsIgnoreCase("ADJP")){
				List<Tree> leaves = node.getLeaves();		//gets all the end leaves of the node labeled as ADJP
				for(Tree leave : leaves)
					buffer.append(leave.label());					
				//System.out.println(buffer);
			}
		}
		return buffer.toString().trim();
	}

	//returns a list of parse tree nodes corresponding to a POS tag, takes as input a Parsed tree and the desired tag	
	private static ArrayList<Tree> extractTag(Tree t, String tag) 
	{
		ArrayList<Tree> wanted = new ArrayList<Tree>();
		if (t.label().value().equals(tag) )
		{
			wanted.add(t);
			for (Tree child : t.children())
			{
				ArrayList<Tree> temp = new ArrayList<Tree>();
				temp=extractTag(child, tag);
				if(temp.size()>0)
				{
					int o =-1;
					o = wanted.indexOf(t);
					if(o!=-1)
						wanted.remove(o);
				}
				wanted.addAll(temp);
			}
		}
		else
			for (Tree child : t.children())
				wanted.addAll(extractTag(child, tag));
		return wanted;
	}

	public static Annotation getAnnotation(String text){
		Annotation annotation = new Annotation(text);
		pipeline.annotate(annotation); 		
		return annotation;
	}

	public static Map<String, List<String>> getTagText(String text, List<String> tags){
		Map<String, List<String>> tagTextMap = new HashMap<String, List<String>>();
		Annotation annotation = new Annotation(text);		
		pipeline.annotate(annotation); 
		List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
		if (sentences != null && sentences.size() > 0) {
			CoreMap sentence = sentences.get(0);
			Tree tree = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
			tree.taggedYield();
			for(String tag : tags){
				ArrayList<Tree> tagTexts = extractTag(tree, tag);
				List<String> list = tagTextMap.get(tag);
				if(list == null) {			
					tagTextMap.put(tag, new ArrayList<String>());			
					for(Tree tagTextTree : tagTexts){
						String tagTextString = Sentence.listToString(tagTextTree.yield());
						tagTextMap.get(tag).add(tagTextString);
					}		
				}		
			}
		}
		return tagTextMap;
	}
	

	public static List<TaggedWord> getTagText(String text){
		Annotation annotation = new Annotation(text);		
		pipeline.annotate(annotation); 
		
		List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
		if (sentences != null && sentences.size() > 0) {
			CoreMap sentence = sentences.get(0);
			Tree tree = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
			ArrayList<TaggedWord> taggedYield = tree.taggedYield();
			return taggedYield;
		}
		return null;
	}

	public static Map<String, List<String>> getTagText(Annotation annotation, List<String> tags){
		Map<String, List<String>> tagTextMap = new HashMap<String, List<String>>();
		List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
		if (sentences != null && sentences.size() > 0) {
			CoreMap sentence = sentences.get(0);
			Tree tree = sentence.get(TreeCoreAnnotations.TreeAnnotation.class);
			for(String tag : tags){
				ArrayList<Tree> tagTexts = extractTag(tree, tag);
				List<String> list = tagTextMap.get(tag);
				if(list == null) {			
					tagTextMap.put(tag, new ArrayList<String>());			
					for(Tree tagTextTree : tagTexts){
						String tagTextString = Sentence.listToString(tagTextTree.yield());
						tagTextMap.get(tag).add(tagTextString);
					}		
				}		
			}
		}
		return tagTextMap;
	}

	//breaks input text into sentences
	public static List<String> getSentences(String text){

		// create an empty Annotation just with the given text	
		List<String> sentences = new ArrayList<String>();

		Annotation document = new Annotation(text);   
		//each text is treated as an Annotation type in order to perform any Stanford NLP task on it

		// run all Annotators on this text
		sentPipeline.annotate(document);

		// these are all the sentences in this document
		// a CoreMap is essentially a Map that uses class objects as keys and has values with custom types
		List<CoreMap> sents = document.get(SentencesAnnotation.class);
		for(CoreMap sentence: sents) 		{
			//System.out.println(sentence);
			sentences.add(sentence.toString());
		}

		return sentences;
	}

	//returns an Arraylist of clauses present in the input sentence
	public static List<String> getClauses(String sentence){
		List<String> clauses = new ArrayList<String>();		
		String clauseTag = "S";
		Annotation annotation = new Annotation(sentence);
		pipeline.annotate(annotation);
		List<CoreMap> sentences = annotation.get(CoreAnnotations.SentencesAnnotation.class);
		String tokenizedSentence = sentence;
		if (sentences != null && sentences.size() > 0) {
			CoreMap sent = sentences.get(0);
			Tree tree = sent.get(TreeCoreAnnotations.TreeAnnotation.class);
			tokenizedSentence = Sentence.listToString(tree.yield());
			ArrayList<Tree> tagTexts = extractTag(tree, clauseTag);		
			for(Tree tagTextTree : tagTexts){
				String clause = Sentence.listToString(tagTextTree.yield());
				clauses.add(clause);
			}
		}

		//it offers , like , ichat , photobooth , and more
		//text = "i even got my teenage son one, because of the features that it offers, like, ichat, photobooth,  and more!";

		String remainingClause = new String(tokenizedSentence);
		for(String clause : clauses)
			remainingClause = remainingClause.replace(clause, "------");
		String[] splits = remainingClause.split("------");
		for(String split : splits){
			if(!split.trim().matches("\\W*"))			
				clauses.add(split.trim());
		}

		return clauses;
	}





	public static void typedDependencies(String text){
		//		LexicalizedParser lp = LexicalizedParser.loadModel("edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz");
		////		lp.apply(sent);
		////		lp.apply(sent).pennPrint();
		//		
		//		TreebankLanguagePack tlp = lp.getOp().langpack();
		//		Tokenizer<? extends HasWord> toke = tlp.getTokenizerFactory().getTokenizer(new StringReader(text));
		//		List<? extends HasWord> sentence = toke.tokenize();
		//		//lp.apply(sentence).
		//		Set<Dependency<Label,Label,Object>> dependencies = lp.apply(sentence).dependencies();
		//		System.out.println(dependencies);


		// create an empty Annotation just with the given text	
		List<String> sentences = new ArrayList<String>();

		Annotation document = new Annotation(text);   
		//each text is treated as an Annotation type in order to perform any Stanford NLP task on it

		// run all Annotators on this text
		sentPipeline.annotate(document);

		// these are all the sentences in this document
		// a CoreMap is essentially a Map that uses class objects as keys and has values with custom types
		List<CoreMap> sents = document.get(SentencesAnnotation.class);
		for(CoreMap sente: sents){
			Tree tree = sente.get(TreeAnnotation.class);
			// Get dependency tree
			TreebankLanguagePack tlp = new PennTreebankLanguagePack();
			GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
			GrammaticalStructure gs = gsf.newGrammaticalStructure(tree);
			Collection<TypedDependency> td = gs.typedDependenciesCollapsed();
			System.out.println(td);

			Object[] list = td.toArray();
			System.out.println(list.length);
			TypedDependency typedDependency;
			for (Object object : list) {
				typedDependency = (TypedDependency) object;
				//System.out.println("Dependency Name " + typedDependency.dep().nodeString()+ " :: "+ "Node "+typedDependency.reln());
				if (typedDependency.reln().getShortName().equals("something")) {
					//your code
				}
			}
			//System.out.println(sentence);
			sentences.add(sente.toString());
		}	
	}

	public static Map<String, List<String>> getTypedDependencies(String text, List<String> aspects, List<String> typedDepencies) {
		Map<String, List<String>> aspectWithDependentWords = new HashMap<String, List<String>>();
		// create an empty Annotation just with the given text	
		//List<String> sentences = new ArrayList<String>();
		Annotation document = new Annotation(text);   
		// run all Annotators on this text
		sentPipeline.annotate(document);
		List<CoreMap> sents = document.get(SentencesAnnotation.class);
		for(CoreMap sente: sents) 	{
			String sentence = sente.toString();
			Tree tree = sente.get(TreeAnnotation.class);
			TreebankLanguagePack tlp = new PennTreebankLanguagePack();
			GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
			GrammaticalStructure gs = gsf.newGrammaticalStructure(tree);
			Collection<TypedDependency> td = gs.typedDependenciesCollapsed();

			for(String aspect : aspects) {
				if(sentence.contains(aspect)) {
					for(TypedDependency typedD : td){
						if(typedDepencies.contains(typedD.reln().toString())) {
							String typedDString = typedD.toString();
							Matcher matcher = pattern.matcher(typedDString);
							if(matcher.find()){
								String one = matcher.group(1).trim();
								String two = matcher.group(2).trim();
								if(one.equalsIgnoreCase(aspect)){
									if(!aspectWithDependentWords.containsKey(aspect))
										aspectWithDependentWords.put(aspect, new ArrayList<String>());
									aspectWithDependentWords.get(aspect).add(two);
								} else if(two.equalsIgnoreCase(aspect)){
									if(!aspectWithDependentWords.containsKey(aspect))
										aspectWithDependentWords.put(aspect, new ArrayList<String>());
									aspectWithDependentWords.get(aspect).add(one);										
								}
							}
						}
					}
				}
			}
		}
		return aspectWithDependentWords;
	}

	public static ArrayList<String> getDependencyNames(String text) {	
		//returns an arraylist containing the names of dependencies found in the given sentence.
		Annotation document = new Annotation(text);   
		sentPipeline.annotate(document);
		List<CoreMap> sents = document.get(SentencesAnnotation.class);
		ArrayList<String> dependencies = new ArrayList<String>();

		for(CoreMap sente: sents){
			//String sentence = sente.toString();
			Tree tree = sente.get(TreeAnnotation.class);
			TreebankLanguagePack tlp = new PennTreebankLanguagePack();
			GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
			GrammaticalStructure gs = gsf.newGrammaticalStructure(tree);
			Collection<TypedDependency> td = gs.typedDependenciesCollapsed();

			for(TypedDependency typedD : td){
				String typedDString = typedD.toString();
				Matcher matcher = pattern1.matcher(typedDString);
				if(matcher.find()){
					String dependencyName = matcher.group(1).trim();
					dependencies.add(dependencyName);
				}

			}
		}
		return dependencies ;
	}


	public static ArrayList<String> getDependencyTriples(String text) {	
		//returns an arraylist containing the names of dependencies found in the given sentence.
		Annotation document = new Annotation(text);   
		sentPipeline.annotate(document);
		List<CoreMap> sents = document.get(SentencesAnnotation.class);
		ArrayList<String> dependencies = new ArrayList<String>();

		for(CoreMap sente: sents){
			Tree tree = sente.get(TreeAnnotation.class);
			TreebankLanguagePack tlp = new PennTreebankLanguagePack();
			GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();
			GrammaticalStructure gs = gsf.newGrammaticalStructure(tree);
			Collection<TypedDependency> td = gs.typedDependenciesCollapsed();
			for(TypedDependency typedD : td){
				String typedDString = typedD.toString();
				Matcher matcher = pattern1.matcher(typedDString);
				if(matcher.find()){
					String dependencyName = matcher.group(1).trim();
					String arg1 = matcher.group(2).trim();
					String arg2 = matcher.group(3).trim();
					String triple = dependencyName+"_"+arg1+"_"+arg2;
					dependencies.add(triple);
				}

			}
		}
		return dependencies ;
	}


	public static void main(String[] args) throws IOException {
		//String text = "However, the multi-touch gestures and large tracking area make having an external mouse unnecessary (unless you're gaming).";
		//text = "i even got my teenage son one, because of the features that it offers, like, ichat, photobooth,  and more!";
		//it offers , like , ichat , photobooth , and more
		//		List<String> aspects = new ArrayList<String>();
		//		aspects.add("multi-touch gestures"); aspects.add("tracking"); aspects.add("mouse"); aspects.add("gaming"); aspects.add("external"); 
		//		aspects.add("area");
		//		List<String> typedDependencies = new ArrayList<String>();
		//		typedDependencies.add("nn"); typedDependencies.add("amod"); typedDependencies.add("xcomp");		
		//		Map<String, List<String>> map = getTypedDependencies(text, aspects, typedDependencies);
		//		for(String aspect : aspects){
		//			List<String> dependentWords = map.get(aspect);
		//			System.out.println(aspect + "\t" + dependentWords);
		//		}
		//String text1 =  "Go to the TIPS tab on the homepage and find further information on the following.";
		//List<String> sentences = StanfordDemoWithTD.getSentences(text1);
		//text = "The magnetic plug-in power charging power cord is great ( I even put it to the test by accident ) - excellent innovation!";
		//System.out.println(getClauses(text));
		//String word = new String("finished");
		//System.out.println(lemmatizeWord(word));
		//List<String> clauses = StanfordDemoWithTD.getClauses(text1);
		//for(String clause : clauses){
		//System.out.println(clause);
		//}
		String text = "Please build one lounge that has AC.";
		//List<String> dependencies = new ArrayList<String>();
		List<String> dependencyTriples = new ArrayList<String>();
		//dependencies = StanfordDemoWithTD.getDependencyNames(text);
//		dependencyTriples = StanfordDemoWithTD.getDependencyTriples(text);
//		for(String dependency : dependencyTriples)
//			System.out.println(dependency);

		List<TaggedWord> tagText = StanfordDemoWithTD.getTagText(text);
		for(TaggedWord tagWord : tagText){
			int index = stanTags.indexOf(tagWord.tag());
			System.out.println(tagWord.word() + " : " + tagWord.tag() + " : " + index);
		}
		
	}
}
