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
public class IBMModel2Aligner implements WordAligner {

    private static final long serialVersionUID = 1315751943476440515L;
    public static final String NULL_WORD = "#NULLWORD#";

    public CounterMap<String,String> t;
    public CounterMap<Tuple,Integer> q;

    public class Tuple {
        public Tuple(int a, int b, int c) {
            j = a;
            l = b;
            m = c;
        }
        
        public boolean equals(Object obj) {
            Tuple o = (Tuple) obj;
            return (o.j == j && o.l == l && o.m == m);
        }

        public int hashCode() {
            return j + 1131*l + 1131*1131*m;
        }
        private int j, l, m;
    }

    public Alignment align(SentencePair sentencePair) {
        Alignment alignment = new Alignment();

        int numSourceWords = sentencePair.getSourceWords().size();
        int numTargetWords = sentencePair.getTargetWords().size();

        for (int ti = 0; ti < numTargetWords; ti++) {
            int maxSource = 0;
            double maxProb = 0;
            String target = sentencePair.getTargetWords().get(ti);
            int l = sentencePair.getSourceWords().size(), m = sentencePair.getTargetWords().size();
            Tuple jlm = new Tuple(ti, l, m);
            for (int si = 0; si < numSourceWords; si++) {
                String source= sentencePair.getSourceWords().get(si);
                double prob = t.getCount(source, target) * q.getCount(jlm, si);
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
        IBMModel1Aligner ibm1 = new IBMModel1Aligner();
        ibm1.train(trainingPairs);

        t = new CounterMap<String,String>();
        t = Counters.conditionalNormalize(ibm1.t);

        q = new CounterMap<Tuple, Integer>();
        for (SentencePair p : trainingPairs) {
            int l = p.getSourceWords().size(), m = p.getTargetWords().size();
            for (int j = 0; j < m; j++) {
                Tuple jlm = new Tuple(j, l, m);
                for (int i = 0; i < l; i++) {
                    q.setCount(jlm, i, 1.0);
                }
            }
        }
        q = Counters.conditionalNormalize(q);

        for (int iter = 0; iter < 20; iter++) {
            CounterMap<String,String> CountsT = new CounterMap<String,String>();
            CounterMap<Tuple,Integer> CountsQ = new CounterMap<Tuple,Integer>();
            for (SentencePair p : trainingPairs) {
                int l = p.getSourceWords().size(), m = p.getTargetWords().size();
                for (int j = 0; j < m; j++) {
                    String target = p.getTargetWords().get(j);
                    Tuple jlm = new Tuple(j, l, m);
                    double sumQ = 0;
                    for (int i = 0; i < l; i++) {
                        String source = p.getSourceWords().get(i);
                        sumQ += q.getCount(jlm, i) * t.getCount(source, target);
                   }
                    for (int i = 0; i < l; i++) {
                        String source = p.getSourceWords().get(i);
                        double increment = q.getCount(jlm, i) * t.getCount(source, target) / sumQ;
                        CountsQ.incrementCount(jlm, i, increment);
                        CountsT.incrementCount(source, target, increment);
                    }
                }
            }
            t = Counters.conditionalNormalize(CountsT);
            q = Counters.conditionalNormalize(CountsQ);
        }
    }
}
