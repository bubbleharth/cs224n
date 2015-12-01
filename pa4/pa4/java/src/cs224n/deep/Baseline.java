package cs224n.deep;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.List;
import java.util.Map;


public class Baseline {
  private static final Map<String, String> wordMap = new HashMap<>();

  // Unambiguous training
  private static void train(List<Datum> trainData) {
    for (Datum datum : trainData) {
      if (wordMap.containsKey(datum.word)) {
        if (wordMap.get(datum.word).equals(datum.label)) {
          continue;
        } else {
          wordMap.put(datum.word, "O");
        }
      } else {
        wordMap.put(datum.word, datum.label);
      }
    }
  }

  public void test(List<Datum> testData, String name) throws IOException {
    String filename = String.format("%s.out", name);
    PrintWriter out = new PrintWriter(new BufferedWriter(new FileWriter(filename)));
    for (Datum datum : testData) {
      String word = datum.word;
      String gold = datum.label;
      String predicted;
      if (wordMap.containsKey(word)) {
        predicted = wordMap.get(word);
      } else {
        predicted = "O";
      }
      out.println(word + "\t" + gold + "\t" + predicted);
    }
    out.close();
  }

  public static void main(String[] args) throws IOException {
    if (args.length < 1) {
      System.out.println("USAGE: java -cp classes Baseline ../data");
      return;
    }
    Baseline baseline = new Baseline();
    String dataDir = args[0];
    FeatureFactory.readWordVectors(dataDir + "/wordVectors.txt");
    FeatureFactory.initializeVocab(dataDir + "/vocab.txt");
    List<Datum> trainData = FeatureFactory.readTrainData(dataDir + "/train");
    List<Datum> finalTest = FeatureFactory.readTestData(dataDir + "/dev");

    train(trainData);
    baseline.test(trainData, "train");
    baseline.test(finalTest, "dev");
  }
}