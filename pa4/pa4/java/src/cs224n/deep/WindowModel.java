package cs224n.deep;
import java.lang.*;
import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;


import java.text.*;

public class WindowModel {

	protected SimpleMatrix L, Wout;

	/* Weights except for the final layer */
	protected SimpleMatrix [] W;

	/* Final layer of MAXENT weights */
	protected SimpleMatrix U;
	
	public int windowSize,wordSize;
	
	/* Number of hidden layers */
	protected int numOfHiddenLayer;

	/* Size of each hidden layers */
	protected int[] hiddenSize;
	
	/* Learning rate */
	protected double alpha;
	
	/* Regularization constant */
	protected double C;

	//Single hidden layer model
	public WindowModel(int windowSize, int hiddenSize, double lr, double reg){
		this.windowSize = windowSize;
		this.wordSize = 50;
		this.numOfHiddenLayer = 1;
		this.W = new SimpleMatrix[numOfHiddenLayer];
		this.hiddenSize = new int[numOfHiddenLayer];
		this.hiddenSize[0] = hiddenSize;
		this.alpha = lr;
		this.C = reg;
	}
	
	//multilayer model
	public WindowModel(int windowSize, int [] hiddenSize, double lr, double reg) {
		this.windowSize = windowSize;
		this.wordSize = 50;
		this.numOfHiddenLayer = hiddenSize.length;
		this.W = new SimpleMatrix[numOfHiddenLayer];
		this.hiddenSize = hiddenSize;
		this.alpha = lr;
		this.C = reg;
	}
	

	/**
	 * Initializes the weights randomly. 
	 */
	public void initWeights(){
		Random rand = new Random();
		L = FeatureFactory.getWordVectors();
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
			
			// Record next fan-out
			fanIn = hiddenSize[i];
		}
		
		// Final layer
		epsilon = Math.sqrt(6.0) / Math.sqrt(1+fanIn);
		U = SimpleMatrix.random(1, fanIn+1, -epsilon, epsilon, rand);
		U.set(0, 0.0);
	}


	/**
	 * Simplest SGD training 
	 */
	public void train(List<Datum> _trainData ){
		//	TODO
	}

	
	public void test(List<Datum> testData){
		// TODO
	}
	
}
