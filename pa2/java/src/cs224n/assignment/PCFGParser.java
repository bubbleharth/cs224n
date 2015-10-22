package cs224n.assignment;

import cs224n.ling.Tree;
import java.util.*;

import cs224n.util.*;

/**
 * The CKY PCFG Parser you will implement.
 */
public class PCFGParser implements Parser {
    private Grammar grammar;
    private Lexicon lexicon;
    private Interner interner = new Interner();

    private ArrayList<IdentityHashMap<Object, Double>> score;
    private ArrayList<IdentityHashMap<Object, Triplet<Integer, String, String>>> back;

    public void train(List<Tree<String>> trainTrees) {
        // TODO: before you generate your grammar, the training trees
        // need to be binarized so that rules are at most binary
        System.out.println("Staring Training Process ......");

        //binarize trees
        ArrayList<Tree<String>> tree_list = new ArrayList<Tree<String>>();
        for (Tree<String> tree : trainTrees) {
            tree_list.add(TreeAnnotations.annotateTree(tree));
        }

        lexicon = new Lexicon(tree_list);
        grammar = new Grammar(tree_list);
        System.out.println(grammar);
    }
    //convert 2d array into 1 d array
    private int getIndex(int x, int y){
        return x + (y*(y-1)/2);
    }

    public Tree<String> getBestParse(List<String> sentence) {

        //CKY Algorithm
        int N = sentence.size() + 1;
        //score = new double[#words+1][#words+1][#nonterms]
        score = new ArrayList<IdentityHashMap<Object, Double>>(N*(N+1)/2);
        //back new Triple[#words+1][#words+1][#nonterms]]
        back = new ArrayList<IdentityHashMap<Object, Triplet<Integer, String, String>>>();
        initalizeDataStructures(N);
        firstPass(sentence);
        return null;
    }

    private void initalizeDataStructures(int length){
        IdentityHashMap<Object, Double> s;
        IdentityHashMap<Object, Triplet<Integer, String, String>> b;
        for (int i = 0; i <= length * (length+1)/2; i++){
            s = new IdentityHashMap<Object, Double> ();
            b = new IdentityHashMap<Object, Triplet<Integer, String, String>> ();
            score.add(s);
            back.add(b);
        }
    }
    //go along the first diagonal
    private void firstPass(List<String> sentence){
        IdentityHashMap<Object, Double> s;
        IdentityHashMap<Object, Triplet<Integer, String, String>> b;
        for (int i = 0; i < sentence.size(); i++){
            int index = getIndex(i, i+1);
            s = score.get(index);
            b = back.get(index);

            for (String binTag: lexicon.getAllTags()){
                s.put(interner.intern(binTag), lexicon.scoreTagging(sentence.get(i), binTag));
                b.put(interner.intern(binTag), new Triplet<Integer, String, String>(-1, sentence.get(i), null));
            }

            //handle unaries
            handleUnaries(s,b);

            score.set(index, s);
            back.set(index, b);
        }
    }

    private void handleUnaries(IdentityHashMap<Object, Double> s,
                                IdentityHashMap<Object, Triplet<Integer, String, String>> b){
        Boolean changed = true;
        while (changed) {
            changed = false;
            for (Object binary: s.keySet()){
                List<Grammar.UnaryRule> unaries = grammar.getUnaryRulesByChild(binary.toString());
                for (Grammar.UnaryRule unary : unaries){
                    double probability = s.get(binary) * unary.getScore();
                    if (!s.containsKey(interner.intern(unary.getParent()))){
                        changed = true;
                        s.put(interner.intern(unary.getParent()), probability);
                        b.put(interner.intern(unary.getParent()), new Triplet<Integer, String, String>(-1, unary.getChild(), null));
                    }
                    else if (probability > s.get(interner.intern(unary.getParent()))){
                        changed = true;
                        s.put(interner.intern(unary.getParent()), probability);
                        b.put(interner.intern(unary.getParent()), new Triplet<Integer, String, String>(-1, unary.getChild(), null));

                    }
                }
            }
        }

    }

    private void cyk(List<String>sentence){

    }
}
