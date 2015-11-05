package cs224n.corefsystems;

import java.util.Collection;
import java.util.List;
import java.util.ArrayList;


import cs224n.coref.ClusteredMention;
import cs224n.coref.Document;
import cs224n.coref.*;
import cs224n.util.Pair;

import cs224n.coref.Mention;
import cs224n.util.*;

public class BetterBaseline implements CoreferenceSystem {

    CounterMap<String, String> corefHeads = null;
	@Override
	public void train(Collection<Pair<Document, List<Entity>>> trainingData) {
        corefHeads = new CounterMap<String, String> ();
        for (Pair<Document, List<Entity>> data : trainingData){
            Document document = data.getFirst();
            List<Entity> clusters = data.getSecond();
            //for each cluster, increment counter for headword pairs together
            for (Entity e: clusters){
                for (Pair<Mention, Mention> mentionPair: e.orderedMentionPairs()){
                    String first_word = mentionPair.getFirst().headWord();
                    String second_word = mentionPair.getSecond().headWord();
                    corefHeads.incrementCount(first_word, second_word, 1.0);
                }
            }
        }
    }

	@Override
	public List<ClusteredMention> runCoreference(Document doc) {
		// TODO Auto-generated method stub
        ArrayList<ClusteredMention> clusters = new ArrayList<ClusteredMention>();
        for (Mention m : doc.getMentions()) {
            if (m.gloss().equals("God the Protector")) {
            System.out.println(m.sentence.parse);
            System.out.println(m.parse);
            System.out.println(m.beginIndexInclusive + " "+m.endIndexExclusive);
            System.out.println(m.gloss());
            }
            clusters.add(m.markSingleton());
        }

	    return clusters;
	}
}
