package cs224n.corefsystems;

import java.util.Collection;
import java.util.List;
import java.util.ArrayList;


import cs224n.coref.ClusteredMention;
import cs224n.coref.Document;
import cs224n.coref.*;
import cs224n.util.Pair;

import cs224n.util.*;
import java.util.*;

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
        ArrayList<ClusteredMention> mentions = new ArrayList<ClusteredMention>();
        Map <String, Entity> clusters = new HashMap <String, Entity>();
        for (Mention mention: doc.getMentions()) {
            String head = mention.getHead();
            if (clusters.containsKey(head)){
                mentions.add(mention.markCoreferent(clusters.get(head)));
            }
            else{
                Set<String> corefs = corefHeads.getCounter(head).keySet();
                boolean added = false;
                for (String s : corefs) {
                    if (corefHeads.getCount(head,s) >= 1.0) {
                        if (clusters.containsKey(s)){
                            mentions.add(mention.markCoreferent(clusters.get(s)));
                            added = true;
                            break;
                        }
                    }
                }
                if (!added){
                    ClusterMention newCluster = mention.markSingleton();
                    mentions.add(newCluster);
                    clusters.put(head, newCluster.entity);
                }
            }

        }

	    return clusters;
	}
}
