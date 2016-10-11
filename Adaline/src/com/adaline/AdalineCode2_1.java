package com.adaline;

import Jama.Matrix;

public class AdalineCode2_1 {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
				
		// matrix for input x(i)
		int xrow = 8;
		int xcol = 3;
		double[][] x1 = {
		
		{1,1,1},
		{1,2,0},
		{2,-1,-1},
		{2,0,0},
		{-1,2,-1},
		{-2,1,1},
		{-1,-1,-1},
		{-2,-2,0}
			};
		
		Matrix x = new Matrix(xrow,xcol);
		for(int i=0;i<xrow;i++)
			for(int j=0;j<xcol;j++)
				x.set(i, j, x1[i][j]);
		
		// Output result from 2 output neurons
		double[][] t1 = {
				
				{-1,-1},
				{-1,-1},
				{-1,1},
				{-1,1},
				{1,-1},
				{1,-1},
				{1,1},
				{1,1}
					};
		int trow = 8;
		int tcol = 2;
		Matrix t = new Matrix(trow,tcol);
		for(int i=0;i<trow;i++)
			for(int j=0;j<tcol;j++)
				t.set(i, j, t1[i][j]);
		// Weights of the input neuron 
		double[][] w1 = {
				
				{0,0,0},
				{0,0,0}
				   };
		int wrow = 2;
		int wcol = 3;
		Matrix w = new Matrix(wrow,wcol);
		for(int i=0;i<wrow;i++)
			for(int j=0;j<wcol;j++)
				w.set(i, j, w1[i][j]);
		
		// Bias of Output Neuron
		double[][] b1 = {
				{0,0},
				{0,0}
		};
		int brow = 1;
		int bcol = 2;
		Matrix b = new Matrix(brow,bcol);
		for(int i=0;i<brow;i++)
			for(int j=0;j<bcol;j++)
				b.set(i, j, b1[i][j]);
		
		// Alpha - learning rate
		double alpha = 1;
		
		int steps = 100;
		int i=0;	// To keep track of stpes
		Matrix Wnew,Wold,Bnew,Bold;
		Wnew = w;
		Bnew = b;
		while(i<steps){
			
			alpha = alpha - alpha/(i+4);		// Delta is Ok and able to classify
			//alpha = alpha - alpha/2;			// Fail to classify
			//alpha = 1/(i+1);					// Not able to classify
			//alpha = 0.1;					//Large delta but able to classify
			//alpha = 0.001;					//Best delta and able to classify
			//alpha = 0.02;
			//alpha = alpha*0.3;			//Not able to classify
			Wold = Wnew;
			Bold = Bnew;
			// x.getMatrix(row start, row end, col start, col end);
			Matrix X = x.getMatrix(i%8,i%8,0,2);
			Matrix Yin = Bold.plus(X.times(Wnew.transpose()));
			Matrix T = t.getMatrix(i%8, i%8, 0, 1);
			Matrix sub = Yin.minus(T);		// X.Wt - T
			Matrix Xt = X.transpose();
			Matrix temp = Xt.times(sub);
			Matrix temp2 = temp.times(2*alpha);
			Wnew = Wold.minus(temp2.transpose());
			Bnew = Bold.minus(sub.times(2*alpha));
			
			System.out.println("Round "+(i+1));
			for(int row=0;row<Wnew.getRowDimension();row++){
				
				for(int col=0;col<Wnew.getColumnDimension();col++){
					
					System.out.print(Wnew.get(row,col)+" ");
				}
				System.out.println();
			}
			System.out.println("Bias: "+Bnew.get(0,0)+" : "+Bnew.get(0,1));
			i++;
		}
		// Delta Calculation
		Matrix delta = new Matrix(2,1);
		for(i=0;i<xrow;i++){
			
			Matrix X = x.getMatrix(i,i,0,2);
			Matrix temp = Wnew.times(X.transpose());
			temp = temp.plus(b.transpose());
			Matrix T = t.getMatrix(i, i, 0, 1);
			temp = temp.minus(T.transpose());
			temp.set(0, 0, Math.pow(temp.get(0, 0),2));
			temp.set(1, 0, Math.pow(temp.get(1, 0), 2));
			delta = delta.plus(temp);
		}
		System.out.println("----DELTA----\n"+delta.get(0, 0)/8+"\n"+delta.get(1, 0)/8);
		
		System.out.println("----CLASSIFICATION----");
		for(i=0;i<xrow;i++){
			
			Matrix X = x.getMatrix(i, i, 0,2);
			System.out.print("[ "+X.get(0,0)+" ,"+X.get(0,1)+" ,"+X.get(0,2)+" ]\t-> ");
			Matrix temp = Wnew.times(X.transpose());
			temp = temp.plus(b.transpose());
			if(temp.get(0,0)<0)
				System.out.print("[ -1,");
			else
				System.out.print("[ 1,");
			if(temp.get(1,0)>=0)
				System.out.println(" 1 ]");
			else
				System.out.println(" -1 ]");
		}
	}
}
