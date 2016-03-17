package edu.insight.unlp.dl4j.sugg.acl2016;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public class Dl4j_ExtendedWVSerializer extends WordVectorSerializer {
	/**
	 * Loads an in memory cache from the given path (sets syn0 and the vocab)
	 *
	 * @param vectorsFile the path of the file to load
	 * @return a Pair holding the lookup table and the vocab cache.
	 * @throws FileNotFoundException if the input file does not exist
	 */
	public static Pair<InMemoryLookupTable, VocabCache> loadTxtComposes(File vectorsFile)
			throws FileNotFoundException {
		BufferedReader reader = new BufferedReader(new FileReader(vectorsFile));
		VocabCache cache = new InMemoryLookupCache();

		LineIterator iter = IOUtils.lineIterator(reader);
		String line = null;
		boolean hasHeader = false;
	
		List<INDArray> arrays = new ArrayList<>();
		while (iter.hasNext()) {
			line = iter.nextLine();
			String[] split = line.split("\t");
			String word = split[0].trim();
			VocabWord word1 = new VocabWord(1.0, word);
			cache.addToken(word1);
			cache.addWordToIndex(cache.numWords(), word);
			word1.setIndex(cache.numWords());
			cache.putVocabWord(word);
			INDArray row = Nd4j.create(Nd4j.createBuffer(split.length - 1));
			for (int i = 1; i < split.length; i++) {
				row.putScalar(i - 1, Float.parseFloat(split[i]));
			}
			arrays.add(row);
		}

		INDArray syn = Nd4j.create(new int[]{arrays.size(), arrays.get(0).columns()});
		for (int i = 0; i < syn.rows(); i++) {
			syn.putRow(i, arrays.get(i));
		}

		InMemoryLookupTable lookupTable = (InMemoryLookupTable) new InMemoryLookupTable.Builder()
				.vectorLength(arrays.get(0).columns())
				.useAdaGrad(false).cache(cache)
				.build();
		Nd4j.clearNans(syn);
		lookupTable.setSyn0(syn);

		iter.close();

		return new Pair<>(lookupTable, cache);
	}

	
	   /**
     * Loads an in memory cache from the given path (sets syn0 and the vocab)
     *
     * @param vectorsFile
     *            the path of the file to load\
     * @return
     * @throws FileNotFoundException
     *             if the file does not exist
     */
    public static WordVectors loadTxtVectorsComposes(File vectorsFile)
            throws FileNotFoundException
    {
        Pair<InMemoryLookupTable, VocabCache> pair = loadTxtComposes(vectorsFile);
        return fromPair(pair);
    }

}
