package edu.insight.unlp.dl4j.examples;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.deeplearning4j.util.SerializationUtils;

import edu.stanford.nlp.ling.TaggedWord;

public class CorrectData {

	public static final String DATA_PATH = "/home/kartik/Dropbox/kat/ACL2016/";

	static File p = null;
	static File n = null;

	private static Map<String, List<TaggedWord>> tagMap = new HashMap<String, List<TaggedWord>>();
	private static String tagMapPath = "src/main/resources/suggData/sentTag.map"; 

	static {
		if(new File(tagMapPath).exists()){
			tagMap = SerializationUtils.readObject(new File(tagMapPath));
		}
	}

	public static void trainOrTest(boolean train, String dataDirectory){
		p = new File(FilenameUtils.concat(dataDirectory, "kat_Experiments/" + (train ? "train" : "test") + "/suggestions/") + "/");
		n = new File(FilenameUtils.concat(dataDirectory, "kat_Experiments/" + (train ? "train" : "test") + "/nonsuggestions/") + "/");
	}

	//3782 + 6959 = 10741	
	public static void main(String[] args) {
		boolean train = true;
		int i = 0;
		int howManyNot = 0;
		trainOrTest(train, DATA_PATH);
		for(File file : p.listFiles()){
			try {
				System.out.println(i);
				String review = FileUtils.readFileToString(file).trim();
				i++;
				if(!tagMap.containsKey(review)){
					howManyNot++;
					List<TaggedWord> tagText = StanfordDemoWithTD.getTagText(review);
					tagMap.put(review,  tagText);					
				}

				if(review.trim().equals("")){
					file.delete();
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}

		for(File file : n.listFiles()){
			try {
				System.out.println(i);
				String review = FileUtils.readFileToString(file).trim();
				i++;
				if(!tagMap.containsKey(review)){
					howManyNot++;
					List<TaggedWord> tagText = StanfordDemoWithTD.getTagText(review);
					tagMap.put(review,  tagText);					
				}

				if(review.trim().equals("")){
					file.delete();
				}
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		serialize();
		System.out.println("Total Data Size: " + i);
		System.out.println("Map not contains: " + howManyNot + " items");
		System.out.println("Map: " + tagMap.size());
	}

	public static void serialize(){
		SerializationUtils.saveObject(tagMap, new File(tagMapPath));
	}

}
