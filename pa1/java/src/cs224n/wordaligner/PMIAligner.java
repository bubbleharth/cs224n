package cs224n.wordaligner;  

import cs224n.util.*;
import java.util.List;

/**
 * Simple word alignment baseline model that maps source positions to target 
 * positions along the diagonal of the alignment grid.
 * 
 * IMPORTANT: Make sure that you read the comments in the
 * cs224n.wordaligner.WordAligner interface.
 * 
 * @author Dan Klein
 * @author Spence Green
 */
public class PMIAligner implements WordAligner {

    private static final long serialVersionUID = 1315751943476440515L;
    private static final String NULL_WORD = "#NULLWORD#";

    // TODO: Use arrays or Counters for collecting sufficient statistics
    // from the training data.
    private Counter<String> sourceWordCounts;
    private Counter<String> targetWordCounts;
    private CounterMap<String,String> sourceTargetCounts;

    public Alignment align(SentencePair sentencePair) {
        // Placeholder code below. 
        // TODO Implement an inference algorithm for Eq.1 in the assignment
        // handout to predict alignments based on the counts you collected with train().
        Alignment alignment = new Alignment();

        // YOUR CODE HERE
        int numSourceWords = sentencePair.getSourceWords().size();
        int numTargetWords = sentencePair.getTargetWords().size();
        sentencePair.targetWords.add(NULL_WORD);

        for (int srcIndex = 0; srcIndex < numSourceWords; srcIndex++) {
            int maxTarget = 0;
            double maxProb = 0;
            String srcWord = sentencePair.getSourceWords().get(srcIndex);
            double srcCount = sourceWordCounts.getCount(srcWord);

            for (int t = 0; t < numTargetWords; t++) {
                String targetWord = sentencePair.getTargetWords().get(t);
                double pairCount = sourceTargetCounts.getCount(srcWord, targetWord);
                double targetCount = targetWordCounts.getCount(targetWord);
                double prob = pairCount * 1.0 / targetCount / srcCount;
                if (prob > maxProb) {
                    maxProb = prob;
                    maxTarget = t;
                }
            }
            if (maxTarget != numTargetWords - 1)
                alignment.addPredictedAlignment(maxTarget, srcIndex);
        }

        return alignment;
    }

    public void train(List<SentencePair> trainingPairs) {
        sourceTargetCounts = new CounterMap<String,String>();
        sourceWordCounts = new Counter<String>();
        targetWordCounts = new Counter<String>();
        for (SentencePair p : trainingPairs) {
            p.targetWords.add(NULL_WORD);
            for (String source : p.getSourceWords()) {
                sourceWordCounts.incrementCount(source, 1);
                for (String target : p.getTargetWords()) {
                    sourceTargetCounts.incrementCount(source, target, 1);
                }
            }
            for (String target : p.getTargetWords()) {
                targetWordCounts.incrementCount(target, 1);
            }
        }
    }
}
