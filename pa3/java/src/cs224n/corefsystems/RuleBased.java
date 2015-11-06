package cs224n.corefsystems;

import java.util.Collection;
import java.util.List;

import cs224n.coref.ClusteredMention;
import cs224n.coref.Document;
import cs224n.coref.Entity;
import cs224n.util.Pair;
import cs224n.util.*;
import cs224n.coref.Mention;

public class RuleBased implements CoreferenceSystem {

    CounterMap<String, String> corefHeads = null;
	@Override
	public void train(Collection<Pair<Document, List<Entity>>> trainingData) {
		// TODO Auto-generated method stub
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
		return null;
	}

}
