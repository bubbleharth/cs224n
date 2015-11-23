package cs224n.corefsystems;

import java.util.*;

import cs224n.coref.*;
import cs224n.util.*;

public class RuleBased implements CoreferenceSystem {

    CounterMap<String, String> corefHeads;
    HashMap<Mention, ClusteredMention> mentions;
    Map <String, Entity> clusters;

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
    
	public List<ClusteredMention> runCoreference(Document doc) {
        corefHeads = new CounterMap<String, String> ();
        mentions = new HashMap<Mention, ClusteredMention>();

        for (Mention m : doc.getMentions()) {
            mentions.put(m, m.markSingleton());
        }

        clusterExactMatch(doc);
        clusterHeadMatch(doc);
        clusterLemmaMatch(doc);
        clusterPronounMentions(doc);

        ArrayList<ClusteredMention> ret = new ArrayList<ClusteredMention>();
        for (Mention m : mentions.keySet()) {
            ret.add(mentions.get(m));
        }
        return ret;
    }

    public void clusterExactMatch(Document doc) {
        clusters = new HashMap <String, Entity>();
        for (Mention m : doc.getMentions()) {
            String full = m.gloss().toLowerCase();
            if (clusters.containsKey(full)){
                for (Mention change : mentions.get(m).entity.mentions) {
                    change.corefferentWith = null;
                    mentions.put(change, change.markCoreferent(clusters.get(full)));
                }
            } else {
                clusters.put(full, mentions.get(m).entity);
            }
        }
    }
    public void clusterHeadMatch(Document doc) {
        clusters = new HashMap <String, Entity>();
        for (Mention m : doc.getMentions()) {
            String head = m.headWord().toLowerCase();
            if (clusters.containsKey(head)){
                for (Mention change : mentions.get(m).entity.mentions) {
                    change.corefferentWith = null;
                    mentions.put(change, change.markCoreferent(clusters.get(head)));
                }
            } else {
                clusters.put(head, mentions.get(m).entity);
            }
        }
    }
    public void clusterLemmaMatch(Document doc) {
        clusters = new HashMap <String, Entity>();
        for (Mention m : doc.getMentions()) {
            String lemma = m.sentence.lemmas.get(m.headWordIndex);
            if (clusters.containsKey(lemma)){
                for (Mention change : mentions.get(m).entity.mentions) {
                    change.corefferentWith = null;
                    mentions.put(change, change.markCoreferent(clusters.get(lemma)));
                }
            } else {
                clusters.put(lemma, mentions.get(m).entity);
            }
        }
    }
    public void clusterPronounMentions(Document doc) {
        Mention previous = null;
        for (Mention m : doc.getMentions()) {
            if (m.headToken().nerTag().equals("PERSON")) {
                previous = m;
            }
            if (!Pronoun.isSomePronoun(m.gloss()) || previous == null) continue;
            Pair<Boolean, Boolean> genders = Util.haveGenderAndAreSameGender(previous, m);
            if (!genders.getFirst() || !genders.getSecond()) continue;
            for (Mention change : mentions.get(m).entity.mentions) {
                change.corefferentWith = null;
                mentions.put(change, change.markCoreferent(mentions.get(previous).entity));
            }
        }
    }
}
