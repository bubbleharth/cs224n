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

    private ArrayList<HashMap<Object, Double>> score;
    private ArrayList<HashMap<Object, Triplet<Integer, String, String>>> back;

    public void train(List<Tree<String>> trainTrees) {
        System.out.println("Staring Training Process ......");

        ArrayList<Tree<String>> tree_list = new ArrayList<Tree<String>>();
        for (Tree<String> tree : trainTrees) {
            tree_list.add(TreeAnnotations.annotateTree(tree));
        }

        lexicon = new Lexicon(tree_list);
        grammar = new Grammar(tree_list);
    }
    private int getIndex(int x, int y) {
        return x + (y*(y-1)/2);
    }

    public Tree<String> getBestParse(List<String> sentence) {
        int N = sentence.size() + 1;
        score = new ArrayList<HashMap<Object, Double>>(N*(N+1)/2);
        back = new ArrayList<HashMap<Object, Triplet<Integer, String, String>>>(N*(N+1)/2);
        initalizeDataStructures(N);

        firstPass(sentence);
        mainCYK(sentence);

        Tree<String> parseTree = new Tree<String>("ROOT");
        addingRules(0, sentence.size(), parseTree, "ROOT");
        return TreeAnnotations.unAnnotateTree(parseTree);
    }

    private void initalizeDataStructures(int length) {
        HashMap<Object, Double> s;
        HashMap<Object, Triplet<Integer, String, String>> b;
        for (int i = 0; i <= length * (length+1)/2; i++) {
            s = new HashMap<Object, Double> ();
            b = new HashMap<Object, Triplet<Integer, String, String>> ();
            score.add(s);
            back.add(b);
        }
    }

    private void firstPass(List<String> sentence) {
        HashMap<Object, Double> s;
        HashMap<Object, Triplet<Integer, String, String>> b;
        for (int i = 0; i < sentence.size(); i++) {
            int index = getIndex(i, i+1);
            s = score.get(index);
            b = back.get(index);

            for (String binTag: lexicon.getAllTags()) {
                s.put(binTag, lexicon.scoreTagging(sentence.get(i), binTag));
                b.put(binTag, new Triplet<Integer, String, String>(-1, sentence.get(i), null));
            }
            handleUnaries(s, b);

            score.set(index, s);
            back.set(index, b);
        }
    }

    private void handleUnaries(HashMap<Object, Double> s,
                                HashMap<Object, Triplet<Integer, String, String>> b) {
        Boolean changed = true;
        while (changed) {
            changed = false;
            Set <Object> keys = new HashSet<Object> (s.keySet());
            for (Object binary : keys) {
                if (s.get(binary) <= 0) continue;
                List<Grammar.UnaryRule> unaries = grammar.getUnaryRulesByChild(binary.toString());
                for (Grammar.UnaryRule unary : unaries) {
                    double prob = s.get(binary) * unary.getScore();
                    if (!s.containsKey(unary.getParent()) || prob > s.get(unary.getParent())) {
                        changed = true;
                        s.put(unary.getParent(), prob);
                        b.put(unary.getParent(), new Triplet<Integer, String, String>(-1, unary.getChild(), null));

                    }
                }
            }
        }
    }

    private void mainCYK(List<String>sentence) {
        HashMap<Object, Double> s;
        HashMap<Object, Triplet<Integer, String, String>> b;
        for (int span = 2; span <= sentence.size(); span++) {
            for (int i = 0; i <= sentence.size() - span; i++) {
                int j = i + span;
                int index = getIndex(i, j);
                s = score.get(index);
                b = back.get(index);
                for (int k = i + 1; k < j; k++) {
                    HashMap<Object, Double> left_scores = score.get(getIndex(i, k));
                    HashMap<Object, Double> right_scores = score.get(getIndex(k, j));
                    for (Object tags : left_scores.keySet()) {
                        for (Grammar.BinaryRule binary : grammar.getBinaryRulesByLeftChild(tags.toString())) {
                            if (!right_scores.containsKey(binary.getRightChild())) continue;
                            double score_left = left_scores.get(binary.getLeftChild());
                            double score_right = right_scores.get(binary.getRightChild());
                            double prob = score_left* score_right * binary.getScore();

                            if (!s.containsKey(binary.getParent()) || prob > s.get(binary.getParent())) {
                                s.put(binary.getParent(), prob);
                                b.put(binary.getParent(), new Triplet<Integer, String, String>(k, binary.getLeftChild(), binary.getRightChild()));
                            }
                        }
                    }
                }
                handleUnaries(s, b);
                score.set(index, s);
                back.set(index, b);
            }
        }
    }

    //recurisvely add rules to the tree
    public void addingRules (int start, int end, Tree<String> parsed, String previous) {
        int index = getIndex(start, end);
        Triplet<Integer, String, String> rule = back.get(index).get(previous);
        List<Tree<String>> leaves = new ArrayList<Tree<String>> ();
        if (rule == null) {
            return;
        } else if (rule.getFirst() != -1) {
            Tree <String> leftSide = new Tree<String>(rule.getSecond());
            addingRules(start, rule.getFirst(), leftSide, rule.getSecond());
            leaves.add(leftSide);

            Tree <String> rightSide = new Tree<String>(rule.getThird());
            addingRules(rule.getFirst(), end, rightSide, rule.getThird());
            leaves.add(rightSide);
        } else {
            Tree<String> leaf = null;
            if (previous.equals(rule.getSecond())) {
                leaf= new Tree<String> (rule.getSecond());
            } else {
                leaf = new Tree<String> (rule.getSecond());
                addingRules(start, end, leaf, rule.getSecond());
            }
            leaves.add(leaf);
        }
       parsed.setChildren(leaves);
    }
}
