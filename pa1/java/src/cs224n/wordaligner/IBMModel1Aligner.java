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
public class IBMModel1Aligner implements WordAligner {

    private static final long serialVersionUID = 1315751943476440515L;
    private static final String NULL_WORD = "#NULLWORD#";

    public CounterMap<String,String> t;

    public Alignment align(SentencePair sentencePair) {
        Alignment alignment = new Alignment();

        int numSourceWords = sentencePair.getSourceWords().size();
        int numTargetWords = sentencePair.getTargetWords().size();

        for (int ti = 0; ti < numTargetWords; ti++) {
            int maxSource = 0;
            double maxProb = 0;
            String target = sentencePair.getTargetWords().get(ti);
            for (int si = 0; si < numSourceWords; si++) {
                String source= sentencePair.getSourceWords().get(si);
                double prob = t.getCount(source, target);
                if (prob > maxProb) {
                    maxProb = prob;
                    maxSource = si;
                }
            }
            if (maxSource != sentencePair.getSourceWords().size()) 
                alignment.addPredictedAlignment(ti, maxSource);
        }

        return alignment;
    }

    public void train(List<SentencePair> trainingPairs) {
        t = new CounterMap<String,String>();
        for (SentencePair p : trainingPairs) {
            for (String source : p.getSourceWords()) {
                for (String target : p.getTargetWords()) {
                    t.setCount(source, target, 1.0);
                }
            }
        }
        t = Counters.conditionalNormalize(t);

        for (int iter = 0; iter < 10; iter++) {
            CounterMap<String,String> Counts = new CounterMap<String,String>();
            for (SentencePair p : trainingPairs) {
                for (String target : p.getTargetWords()) {
                    double sumPost = 0;
                    for (String source : p.getSourceWords()) {
                        sumPost += t.getCount(source, target);
                    }
                    for (String source : p.getSourceWords()) {
                        Counts.incrementCount(source, target, t.getCount(source, target) / sumPost);
                    }
                }
            }
            t = Counters.conditionalNormalize(Counts);
        }
    }
}
