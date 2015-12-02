package cs224n.deep;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;
import java.io.*;

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

	/*multi-layer model */
	public WindowModel(int windowSize, int [] hiddenSize, double lr, double reg) {
		this.windowSize = windowSize;
		this.wordSize = 50;
		this.numOfHiddenLayer = hiddenSize.length;
		this.W = new SimpleMatrix[numOfHiddenLayer];
		this.hiddenSize = hiddenSize;
		this.alpha = lr;
		this.lambda = reg;
		System.out.println("Parameters:");
		System.out.println("-------------------------");
		System.out.println("Window Size: " + windowSize);
		System.out.println("Word Size: " + wordSize);
		System.out.println("Number of Hidden Layers: " + numOfHiddenLayer);
		System.out.println("Learning Rate: " + lr);
		System.out.println("Regularization: " + reg);
		
	}

	 //Simple math functions that will need to be used
	  private SimpleMatrix softmax(SimpleMatrix M) {
		  SimpleMatrix soft = new SimpleMatrix(M.numRows(), 1);
		  double sum = 0;
		  for (int i = 0; i < M.numRows(); i++) {
			  sum += Math.exp(M.get(i, 0));
		  }
		  for (int i = 0; i < M.numRows(); i++) {
			  soft.set(i, 0, Math.exp(M.get(i, 0)) / sum);
		  }
		  return soft;
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
		L = FeatureFactory.allVecs;
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
		U = SimpleMatrix.random(5, fanIn+1, -epsilon, epsilon, rand);
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

	private List<SimpleMatrix> makeGroundTruth(List<Datum> data){
		List<SimpleMatrix> labels = new ArrayList<SimpleMatrix>();
		for (int i = 0; i < data.size(); ++i) {
			labels.add(LABEL_VECTORS.get(data.get(i).label));
		}
		return labels;
	}
	//ground truth labels
	private static final String[] LABELS = {"O", "LOC", "MISC", "ORG", "PER"};
	private static final Map<String, SimpleMatrix> LABEL_VECTORS;
	static {
		LABEL_VECTORS = new HashMap<>();
		for (int i = 0; i < LABELS.length; i++) {
			SimpleMatrix m = new SimpleMatrix(LABELS.length, 1);
		    m.set(i, 0, 1);
		    LABEL_VECTORS.put(LABELS[i], m);
		    }
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
		temp = softmax(U.mult(addBiasTerm(temp)));
		return temp;
	}
	
	
	public double getCost(SimpleMatrix X) {
		SimpleMatrix output = batchFeedForward(X);
		int numOfSamples = X.numCols();
		
		double cost = 0.0;
		//cross entropy.
	    for (int i = 0; i < X.numRows(); i++) {
	        cost += X.get(i, 0) * Math.log(output.get(i, 0));
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
	
	private SimpleMatrix[] backpropGrad(SimpleMatrix batch, SimpleMatrix label) {
		SimpleMatrix [] activation = new SimpleMatrix[numOfHiddenLayer+2];
		SimpleMatrix [] input_vecs = new SimpleMatrix[numOfHiddenLayer+2];

		
		// Forward propagation. Reminder that we are passing inputs without the bias term added 
		for (int i = 0; i < numOfHiddenLayer+2; ++i) {
			// Input layer
			if (i == 0) {
				input_vecs[i] = batch;
				activation[i] = addBiasTerm(input_vecs[i]);
			}
			// Output layer
			else if (i == numOfHiddenLayer+1) {
				input_vecs[i] = U.mult(activation[i-1]);

				activation[i] = softmax(input_vecs[i]);
			}
			// Middle layer
			else {
				input_vecs[i] = W[i-1].mult(activation[i-1]);
				activation[i] = addBiasTerm(tanh(input_vecs[i]));
			}
		}
		
		// Initialize a list of gradient matrices
		SimpleMatrix[] gradients = new SimpleMatrix[numOfHiddenLayer+2];
		for (int i = 0; i < gradients.length; ++i) {
			// Gradient for L
			if (i == 0) {
				gradients[i] = new SimpleMatrix(batch.numRows(), 1);
			}
			// Gradient for U
			else if (i == numOfHiddenLayer+1) {
				gradients[i] = new SimpleMatrix(U.numRows(), U.numCols());
			}
			// Gradient for W
			else {
				gradients[i] = new SimpleMatrix(W[i-1].numRows(), W[i-1].numCols());
			}
		}
		
		int numCols = batch.numCols();
		SimpleMatrix Error = activation[numOfHiddenLayer+1].minus(label);
		
		// For each instance in this batch
		//Starting from the back
		for (int m = 0; m < numCols; ++m) {
			SimpleMatrix Delta = Error.extractVector(false, m);
		
			for (int i = numOfHiddenLayer+1; i >= 1; --i) {
				gradients[i] = gradients[i].plus(Delta.mult(activation[i-1].extractVector(false, m).transpose()));
				//Output Layer
				if (i == numOfHiddenLayer+1) {
					Delta = U.transpose().mult(Delta);
					Delta = Delta.extractMatrix(1, Delta.numRows(), 0, Delta.numCols());
					Delta = Delta.elementMult(tanhDerivative(input_vecs[i-1].extractVector(false, m)));
				}
				else if (i == 1) {
					Delta = W[i-1].transpose().mult(Delta);
					Delta = Delta.extractMatrix(1, Delta.numRows(), 0, Delta.numCols());
				}
				// Hidden layer
				else {
					Delta = W[i-1].transpose().mult(Delta);
					Delta = Delta.extractMatrix(1, Delta.numRows(), 0, Delta.numCols());
					Delta = Delta.elementMult(tanhDerivative(input_vecs[i-1].extractVector(false, m)));
				}
			}
			// Input layer
			gradients[0] = gradients[0].plus(Delta);

		}
		
		// Average and add regularization term
		for (int i = 0; i < gradients.length; ++i) {
			// Gradient for L
			if (i == 0) {
				gradients[i] = gradients[i].divide(numCols);
			}
			// Gradient for U
			else if (i == numOfHiddenLayer+1) {
				SimpleMatrix nobiasU = U.copy();
				double [] arr = new double[nobiasU.numRows()];
				Arrays.fill(arr, 0.0);
				nobiasU.setColumn(0, 0, arr);
				gradients[i] = gradients[i].divide(numCols).plus(nobiasU.scale(lambda / numCols));
			}
			// Gradient for W
			else {
				SimpleMatrix nobiasW = W[i-1].copy();
				double [] arr = new double[nobiasW.numRows()];
				Arrays.fill(arr, 0.0);
				nobiasW.setColumn(0, 0, arr);
				gradients[i] = gradients[i].divide(numCols).plus(nobiasW.scale(lambda / numCols));
			}
		}
		
		return gradients;
	}
	
	/**
	 * Simplest SGD training 
	 */
	public void train(List<Datum> _trainData, int epoch ){
		List<List<Integer>> TrainingX = makeInputContextWindows(_trainData);
		List<SimpleMatrix> TrainingY = makeGroundTruth(_trainData);
		Random random = new Random();
		int numTrain = _trainData.size();
		for (int e = 1; e <= epoch; e++){
			System.out.println("EPOCH: " + e);
			// For each training example
			long seed = System.nanoTime();
			Collections.shuffle(TrainingX, new Random(seed));
			Collections.shuffle(TrainingY, new Random(seed));
			for (int i = 0; i < numTrain; ++i) {
				if (i%10000 ==0) System.out.println("Training Example: " + i);
				SimpleMatrix input = makeInput(TrainingX.get(i));
				SimpleMatrix label = TrainingY.get(i);
				// Compute Gradient
				SimpleMatrix [] G = backpropGrad(input, label);
				
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
		System.out.println("Train statistics");
		try {
			TrainingX = makeInputContextWindows(_trainData);
			PrintWriter pw = new PrintWriter(new File("/Users/Jasper/Documents/cs224n/pa4/pa4/train.out"));
			for (int i = 0; i < TrainingX.size(); ++i) {
				SimpleMatrix input = makeInput(TrainingX.get(i));
				SimpleMatrix response = batchFeedForward(input);
				String output = LABELS[getArgMaxIndex(response)];
				pw.println(FeatureFactory.getTrainData().get(i).word + "\t" + _trainData.get(i).label + "\t" + output + "\n");
			}
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

	}
	public void test(List<Datum> testData){
		List<List<Integer>> TestingX = makeInputContextWindows(testData);
		// Evaluate test statistics
		System.out.println("Test statistics");
		try {
			PrintWriter pw = new PrintWriter(new File("/Users/Jasper/Documents/cs224n/pa4/pa4/test.out"));
			for (int i = 0; i < TestingX.size(); ++i) {
				SimpleMatrix input = makeInput(TestingX.get(i));
				SimpleMatrix response = batchFeedForward(input);
				String output = LABELS[getArgMaxIndex(response)];
				pw.println(FeatureFactory.getTestData().get(i).word + "\t" + testData.get(i).label + "\t" + output + "\n");
			}
			pw.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	private int getArgMaxIndex(SimpleMatrix scores) {
		int idx = 0;
		for (int i = 1; i < scores.numRows(); i++) {
			if (scores.get(i, 0) > scores.get(idx, 0)) idx = i;
		}
		return idx;
	}
}
