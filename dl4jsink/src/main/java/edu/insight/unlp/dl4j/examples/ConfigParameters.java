package edu.insight.unlp.dl4j.examples;


import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

public class ConfigParameters {

	private List<String> listOfClassifiers;
	private Integer noOfAttributes;
	private String outputFile;	
	private Integer noOfFolds;

	public List<String> getListOfClassifiers() {
		return listOfClassifiers;
	}

	public int getNoOfAttributes(){
		return noOfAttributes;	
	}

	public ConfigParameters(String path) {
		parameterParser(path);
	}

	private void parameterParser(String path) {
		JSONParser parser = new JSONParser();
		JSONObject jsonObject = null;
		try {
			jsonObject = (JSONObject) parser.parse(new FileReader(path));
			JSONArray classificationArgs = (JSONArray) jsonObject.get("Classifier");			
			listOfClassifiers = new ArrayList<String>();
			for (Object object : classificationArgs) 
				listOfClassifiers.add((String)object);			
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		} catch (ParseException e) {
			e.printStackTrace();
		}
	}

	public String getOutputFile() {
		return outputFile;
	}

	public void setOutputFile(String outputFile) {
		this.outputFile = outputFile;
	}

	public Integer getNoOfFolds() {
		return noOfFolds;
	}

	public void setNoOfFolds(Integer noOfFolds) {
		this.noOfFolds = noOfFolds;
	}
}
