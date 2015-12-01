package cs224n.deep;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;
import org.ejml.simple.*;


public class FeatureFactory {
	private static final int numRows = 100232; // Number of word vectors.
	private static final int numCols = 50; // Number of dimensions.


	private FeatureFactory() {
		
	}
	public static List<Datum> getTrainData() {
		return trainData;
	}
	
	public static List<Datum> getTestData() {
		return testData;
	}
	
	public static HashMap<String, Integer> getDictionary() {
		return wordToNum;
	}


	static List<Datum> trainData;
	/** Do not modify this method **/
	public static List<Datum> readTrainData(String filename) throws IOException {
        if (trainData==null) trainData= read(filename);
        return trainData;
	}

	static List<Datum> testData;
	/** Do not modify this method **/
	public static List<Datum> readTestData(String filename) throws IOException {
        if (testData==null) testData= read(filename);
        return testData;
	}

	private static List<Datum> read(String filename)
			throws FileNotFoundException, IOException {
	    // TODO: you'd want to handle sentence boundaries
		List<Datum> data = new ArrayList<Datum>();
		BufferedReader in = new BufferedReader(new FileReader(filename));
		data.add(new Datum("<s>", "O"));
		for (String line = in.readLine(); line != null; line = in.readLine()) {
			if (line.trim().length() == 0) {
				data.add(new Datum("</s>", "O"));
			    data.add(new Datum("<s>", "O"));
			    continue;
			}
			String[] bits = line.split("\\s+");
			String word = bits[0];
			String label = bits[1];

			Datum datum = new Datum(word.toLowerCase(), label);
			data.add(datum);
		}
		in.close();
		return data;
	}


	// Look up table matrix with all word vectors as defined in lecture with dimensionality n x |V|
	static SimpleMatrix allVecs; //access it directly in WindowModel
	  public static SimpleMatrix readWordVectors(String vecFilename) throws IOException {
		    if (allVecs != null) return allVecs;
		    double[][] mat = new double[numRows][numCols];
		    BufferedReader in = new BufferedReader(new FileReader(vecFilename));
		    int row = 0;
		    for (String line = in.readLine(); line != null; line = in.readLine()) {
		      String[] nums = line.split("\\s+");
		      if (nums.length != numCols) {
		        throw new RuntimeException("dimension does not match");
		      }
		      for (int col = 0; col < numCols; col++) {
		        mat[row][col] = Double.parseDouble(nums[col]);
		      }
		      row++;
		    }
		    if (row != numRows) {
		      throw new RuntimeException("dimension does not match");
		    }
		    in.close();
		    allVecs = new SimpleMatrix(mat);
		    return allVecs;
		  }
	  
	// might be useful for word to number lookups, just access them directly in WindowModel
	public static HashMap<String, Integer> wordToNum = new HashMap<String, Integer>();
	public static HashMap<Integer, String> numToWord = new HashMap<Integer, String>();

	public static HashMap<String, Integer> initializeVocab(String vocabFilename) throws IOException {
	    BufferedReader in = new BufferedReader(new FileReader(vocabFilename));
	    int i = 0;
	    for (String line = in.readLine(); line != null; line = in.readLine()) {
	      String word = line.trim();
	      wordToNum.put(word, i);
	      numToWord.put(i, word);
	      i++;
	    }
	    in.close();
	    return wordToNum;
	  }
}
