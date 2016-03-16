package edu.insight.unlp.dl4j.sugg.acl2016;


import java.io.BufferedReader;

//performs..
//author snegi

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import edu.insight.unlp.otdf.BasicFileTools;
import edu.stanford.nlp.ling.CoreAnnotations;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreeCoreAnnotations;
import edu.stanford.nlp.util.CoreMap;

public class StanfordNLPUtil {


	private static StanfordCoreNLP pipeline;
	public static String stantagsFile = "/home/kartik/git/ytip/classifier/src/main/resources/StanTags";
	public static List<String> stanTags = new ArrayList<String>();
	public static String tagsRegex = "\\d+\\.\\s+([A-Z$]+)";
	public static Pattern tagsPatt = Pattern.compile(tagsRegex);

	private static	Properties props = new Properties();
	static {
		props.put("annotators", "tokenize, ssplit, parse"); //, lemma
		pipeline = new StanfordCoreNLP(props);
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

	public static void main(String[] args) throws IOException {
		String text = "Please build one lounge that has AC.";
		List<TaggedWord> tagText = StanfordNLPUtil.getTagText(text);
		for(TaggedWord tagWord : tagText){
			int index = stanTags.indexOf(tagWord.tag());
			System.out.println(tagWord.word() + " : " + tagWord.tag() + " : " + index);
		}		
	}

}
