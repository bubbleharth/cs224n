package cs224n.corefsystems;

import java.util.Collection;
import java.util.List;
import java.util.*;

import cs224n.coref.ClusteredMention;
import cs224n.coref.Document;
import cs224n.coref.Entity;
import cs224n.util.Pair;

import cs224n.coref.Mention;

public class OneCluster implements CoreferenceSystem {

	@Override
	public void train(Collection<Pair<Document, List<Entity>>> trainingData) {
		// TODO Auto-generated method stub

	}

	@Override
	public List<ClusteredMention> runCoreference(Document doc) {
        //One cluster, define cluster as first mention, everything gets
        //clustered in as well
        ClusteredMention onecluster = null;
        List<ClusteredMention> mentions = new ArrayList<ClusteredMention> ();
        for (Mention mention : doc.getMentions()){
           if (onecluster == null){
               onecluster = mention.markSingleton();
               mentions.add(onecluster);
            }
            else {
               mentions.add(mention.markCoreferent(onecluster.entity));
            }

        }
        return mentions;
	}
}
