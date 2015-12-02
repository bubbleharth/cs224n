package cs224n.deep;

import java.util.*;
import java.io.*;

import org.ejml.simple.SimpleMatrix;


public class NER {
    
    public static void main(String[] args) throws IOException {
	if (args.length < 2) {
	    System.out.println("USAGE: java -cp classes NER ../data/train ../data/dev");
	    return;
	}	    

	// this reads in the train and test datasets
	List<Datum> trainData = FeatureFactory.readTrainData("/Users/Jasper/Documents/cs224n/pa4/pa4/data/train");
	List<Datum> testData = FeatureFactory.readTestData("/Users/Jasper/Documents/cs224n/pa4/pa4/data/dev");
	
	//	read the train and test data
	//TODO: Implement this function (just reads in vocab and word vectors)
//	FeatureFactory.initializeVocab("../data/vocab.txt");
//	SimpleMatrix allVecs= FeatureFactory.readWordVectors("../data/wordVectors.txt");
	FeatureFactory.initializeVocab("/Users/Jasper/Documents/cs224n/pa4/pa4/data/vocab.txt");
	SimpleMatrix allVecs= FeatureFactory.readWordVectors("/Users/Jasper/Documents/cs224n/pa4/pa4/data/wordVectors.txt");
	// initialize model 
	String[] layersStr = "100".split(",");
	int [] layer = new int[layersStr.length];
	for (int i = 0; i < layersStr.length; ++i) {
		layer[i] = Integer.valueOf(layersStr[i]).intValue();
	}
	WindowModel model = new WindowModel(5, layer, 0.01, 0.0001);
	model.initWeights();
	System.out.println("Starting training...");
	model.train(trainData, 10);
	
	model.test(testData);
    }
}