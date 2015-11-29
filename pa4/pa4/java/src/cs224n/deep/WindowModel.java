package cs224n.deep;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;

public class WindowModel {
	/* Word vectors */
	protected SimpleMatrix L;

	/* Weights except for the final layer */
	protected SimpleMatrix [] W;

	/* Final layer of MAXENT weights */
	protected SimpleMatrix U;
	
	/* Context window size */
	protected int windowSize;

	/* Word vector dimensions */
	protected int wordSize;
	
	/* Number of hidden layers */
	protected int numOfHiddenLayer;

	/* Size of each hidden layers */
	protected int[] hiddenSize;
	
	/* Learning rate */
	protected double alpha;
	
	/* Regularization constant */
	protected double lambda;
	
	/* Unseen word placeholder */
	public static final String UNKNOWN = "UUUNKKK";

	/*Single hidden layer model */
	public WindowModel(int windowSize, int hiddenSize, double lr, double reg){
		this.windowSize = windowSize;
		this.wordSize = 50;
		this.numOfHiddenLayer = 1;
		this.W = new SimpleMatrix[numOfHiddenLayer];
		this.hiddenSize = new int[numOfHiddenLayer];
		this.hiddenSize[0] = hiddenSize;
		this.alpha = lr;
		this.lambda = reg;
	}
	
	/*multilayer model */
	public WindowModel(int windowSize, int [] hiddenSize, double lr, double reg) {
		this.windowSize = windowSize;
		this.wordSize = 50;
		this.numOfHiddenLayer = hiddenSize.length;
		this.W = new SimpleMatrix[numOfHiddenLayer];
		this.hiddenSize = hiddenSize;
		this.alpha = lr;
		this.lambda = reg;
	}
	

	 //Simple math functions that will need to be used
	private static SimpleMatrix sigmoid(SimpleMatrix M) {
		SimpleMatrix sig = new SimpleMatrix(M.numRows(), M.numCols());
		for (int i = 0; i < M.numRows(); ++i) {
			for (int j = 0; j < M.numCols(); ++j) {
				sig.set(i, j, 1.0 / (1.0 + Math.exp(-M.get(i, j))));
			}
		}
		return sig;
	}
	
	private static SimpleMatrix tanh(SimpleMatrix M) {
		SimpleMatrix tanh = new SimpleMatrix(M.numRows(), M.numCols());
		for (int i = 0; i < M.numRows(); ++i) {
			for (int j = 0; j < M.numCols(); ++j) {
				tanh.set(i, j, Math.tanh(M.get(i, j)));
			}
		}
		return tanh;
	}
	
	private static SimpleMatrix tanhDerivative(SimpleMatrix M) {
		SimpleMatrix tanhD = new SimpleMatrix(M.numRows(), M.numCols());
		for (int i = 0; i < M.numRows(); ++i) {
			for (int j = 0; j < M.numCols(); ++j) {
				tanhD.set(i, j, 1.0 - Math.tanh(M.get(i, j)) * Math.tanh(M.get(i, j)));
			}
		}
		return tanhD;
	}

	private SimpleMatrix makeInput(List<Integer> win) {
		SimpleMatrix vec = new SimpleMatrix(wordSize * windowSize, 1);
		for (int w = 0; w < win.size(); ++w) {
			vec.insertIntoThis(w * wordSize, 0, L.extractVector(false, win.get(w)));
		}
		return vec;
	}
	/**
	 * Initializes the weights randomly. 
	 */
	public void initWeights(){
		Random rand = new Random();
		L = FeatureFactory.getWordVectors();
		//First layer size
		int fanIn = windowSize * wordSize;
		double epsilon;
		for (int i = 0; i < numOfHiddenLayer; ++i) {
			// Initialize hidden weights to random numbers
			epsilon = Math.sqrt(6.0) / Math.sqrt(hiddenSize[i] + fanIn); 
			W[i] = SimpleMatrix.random(hiddenSize[i], fanIn+1, -epsilon, epsilon, rand);
			
			// Initialize bias terms to zeros
			double [] zeros = new double[hiddenSize[i]];
			Arrays.fill(zeros, 0.0);
			W[i].setColumn(0, 0, zeros);
			fanIn = hiddenSize[i];
		}
		
		// Final layer
		epsilon = Math.sqrt(6.0) / Math.sqrt(1+fanIn);
		U = SimpleMatrix.random(1, fanIn+1, -epsilon, epsilon, rand);
		U.set(0, 0.0);
	}
	
	//get correct input windows with context. output is list of list of integers that
	//correspond to indices in the dictionary
	
	private List<List<Integer>> makeInputContextWindows(List<Datum> data) {
		int radius = windowSize / 2;
		HashMap<String, Integer> dict = FeatureFactory.getDictionary();
		
		List<List<Integer>> contextWindowList = new ArrayList<List<Integer>>();
		
		for (int i = 0; i < data.size(); ++i) {
			Datum instance = data.get(i);
			LinkedList<String> window = new LinkedList<String>();
			window.add(instance.word);
			
			// Expand window
			boolean left = instance.word.equals("<s>");
			boolean right = instance.word.equals("</s>");
			
			// Expand left
			int ll = i-1;
			for (int r = 0; r < radius; ++r) {
				if (left) window.addFirst("<s>");
				else if (ll < 0) window.addFirst("<s>");
				else {
					window.addFirst(data.get(ll).word);
					left = data.get(ll).word.equals("<s>");
				}
				--ll;
			}
			
			// Expand right
			int rr = i+1;
			for (int r = 0; r < radius; ++r) {
				if (right) window.addLast("</s>");
				else if (rr >= data.size()) window.addLast("</s>");
				else {
					window.addLast(data.get(rr).word);
					right = data.get(rr).word.equals("</s>");
				}
				++rr;
			}
			
			// Convert strings to index of dictionary
			ArrayList<Integer> windowIndices = new ArrayList<Integer>();
			for (String word : window) {
				if (dict.containsKey(word)) windowIndices.add(dict.get(word));
				else windowIndices.add(dict.get(UNKNOWN));
			}
			contextWindowList.add(windowIndices);
		}
		return contextWindowList;
	}

	//ground truth labels
	private List<Integer> groundTruthLabels(List<Datum> data) {
		List<Integer> labels = new ArrayList<Integer>();
		for (int i = 0; i < data.size(); ++i) {
			if (data.get(i).label.equals("PERSON")) {
				labels.add(1);
			}
			else labels.add(0);
		}
		return labels;
	}
	
	//add bias term to matrixes
	public static SimpleMatrix addBiasTerm(SimpleMatrix A) {
		SimpleMatrix B = new SimpleMatrix(1, A.numCols());
		B.set(1.0);
		return B.combine(B.numRows(), 0, A);
	}
	
	//Simple multiplication of terms to get the output
	public SimpleMatrix batchFeedForward(SimpleMatrix window) {
		SimpleMatrix temp = window;
		for (int i = 0; i < numOfHiddenLayer; ++i) {
			temp = tanh(W[i].mult(addBiasTerm(temp)));
		}
		temp = sigmoid(U.mult(addBiasTerm(temp)));
		return temp;
	}
	
	
	public double getCost(SimpleMatrix X, SimpleMatrix L) {
		SimpleMatrix output = batchFeedForward(X);
		int numOfSamples = X.numCols();
		
		double cost = 0.0;
		//cross entropy. because it's binary, probability of not getting 'Person' is just 1 - p('Person')
		for (int i = 0; i < numOfSamples; ++i) {
			cost += (L.get(0, i) * Math.log(output.get(0, i)) + (1 - L.get(0, i)) * Math.log(1 - output.get(0, i)));
		}
		
		cost /= -numOfSamples;
		
		// Regularization without bias terms as suggested by prompt
		for (int i = 0; i < W.length; ++i) {
			SimpleMatrix nobiasW = W[i].extractMatrix(0, W[i].numRows(), 1, W[i].numCols());
			cost += lambda / 2 * nobiasW.elementMult(nobiasW).elementSum();
		}
		SimpleMatrix nobiasU = U.extractMatrix(0, U.numRows(), 1, U.numCols());
		cost += lambda / 2 * nobiasU.elementMult(nobiasU).elementSum();
		
		return cost;
	}
	private SimpleMatrix[] backPropGrad(SimpleMatrix batch, SimpleMatrix label) {
		//temp placeholder
		SimpleMatrix [] a = new SimpleMatrix[numOfHiddenLayer+2];
		SimpleMatrix [] z = new SimpleMatrix[numOfHiddenLayer+2];
		
		return z;
	}
	
	/**
	 * Simplest SGD training 
	 */
	public void train(List<Datum> _trainData, int epoch ){
		List<List<Integer>> TrainingX = makeInputContextWindows(_trainData);
		List<Integer> TrainingY = groundTruthLabels(_trainData);
		int numTrain = _trainData.size();
		for (int e = 0; e < epoch; e++){
			
			// Randomly shuffle examples
			long seed = System.nanoTime();
			Collections.shuffle(TrainingX, new Random(seed));
			Collections.shuffle(TrainingY, new Random(seed));
						
			// For each training example
			for (int i = 0; i < numTrain; ++i) {
				SimpleMatrix input = makeInput(TrainingX.get(i));
				SimpleMatrix label = new SimpleMatrix(1, 1);
				label.set(TrainingY.get(i));
				
				// Compute Gradient
				SimpleMatrix [] G = backPropGrad(input, label);
				
				// Update W
				for (int j = 1; j <= numOfHiddenLayer; ++j) {
					W[j-1] = W[j-1].minus(G[j].scale(alpha));
				}
				
				// Update U
				U = U.minus(G[numOfHiddenLayer+1].scale(alpha));
				
				// Update L
				input = input.minus(G[0].scale(alpha));
				List<Integer> wordIdx = TrainingX.get(i);
				for (int idx = 0; idx < wordIdx.size(); ++idx) {
					L.insertIntoThis(0, wordIdx.get(idx), input.extractMatrix(idx * wordSize, (idx+1) * wordSize, 0, 1));
				}
			}
			
		}
		// Evaluate training statistics
		System.out.println("Training statistics");
		evaluateStatistics(TrainingX, TrainingY);
		System.out.println();
	}

	
	public void test(List<Datum> testData){
		List<List<Integer>> TestingX = makeInputContextWindows(testData);
		List<Integer> TestingY = groundTruthLabels(testData);
		// Evaluate test statistics
		System.out.println("Test statistics");
		evaluateStatistics(TestingX, TestingY);
		System.out.println();
	}
	
	private void evaluateStatistics(List<List<Integer>> Data, List<Integer> Label) {
		int numData = Data.size();
		int truePositive = 0, falsePositive = 0, falseNegative = 0, trueNegative = 0;
		for (int i = 0; i < numData; ++i) {
			SimpleMatrix input = makeInput(Data.get(i));
			SimpleMatrix response = batchFeedForward(input);
			
			int result = 0;
			if (response.get(0) > 0.5) result = 1;
			int answer = Label.get(i).compareTo(1) == 0 ? 1 : 0;
			if (result == answer && answer == 1) {
				++truePositive;
			} else if (result == answer && result == 0) {
				++trueNegative;
			} else if (result != answer && result == 1) {
				++falsePositive;
			} else if (result != answer && answer == 1) {
				++falseNegative;
			}
		}
		
		System.out.println("-------------------------------");
		System.out.println("Dataset size: " + numData);
		System.out.println("PERSON Precision: " + (double)truePositive / ((double)(truePositive+falsePositive)));
		System.out.println("PERSON Recall: " + (double)truePositive / ((double)(truePositive+falseNegative)));
		System.out.println("NON-PERSON Precision: " + (double)trueNegative / ((double)(trueNegative+falseNegative)));
		System.out.println("NON-PERSON Recall: " + (double)trueNegative / ((double)(trueNegative+falsePositive)));
		System.out.println("-------------------------------");
	}
	
}
