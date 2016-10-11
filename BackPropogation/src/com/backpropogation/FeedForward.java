
package com.backpropogation;

import Jama.Matrix;

public class FeedForward {

	public static double FofX(double x){
		
		return Math.tanh(x/2);
	}
	public static double dFofX(double x){
		
		return (1-Math.pow(Math.tanh(x/2), 2))/2;
	}
	public static void printMatrix(Matrix m, int row, int col){
		
		for(int i=0;i<row;i++){
			
			System.out.print("[");
			for(int j=0;j<col;j++){
				
				System.out.print(m.get(i, j)+" ");
			}
			System.out.println("]");
		}
	}
	public static void main(String[] args) {
		// TODO Auto-generated method stub

		//hidden layer weights and bias
		double[][] harr = {
				{-0.15, 0.2},	// [ w11 w21 ]
				{0.25, -0.3}		// [ w12 w22 ]
		};
		int hwrow = 2, hwcol = 2;
		Matrix hw = new Matrix(hwrow,hwcol);
		for(int i=0;i<hwrow;i++)
			for(int j=0;j<hwcol;j++){
			
				hw.set(i, j, Math.random()-0.5);
				//hw.set(i, j, 0.3);
			}
		int hbrow = 2, hbcol = 1;
		Matrix hb = new Matrix(hbrow,hbcol);
		for(int i=0;i<hbrow;i++)
			for(int j=0;j<hbcol;j++){

				hb.set(i, j, Math.random()-0.5);
				//hb.set(i, j, 0.4);
			}	
		
		// Weight and Bias for Output neuron
		double[][] oarr = {
				{0.4},
				{0.45}
		};
		int oprow = 2, opcol = 1;
		Matrix w = new Matrix(oprow,opcol);
		for(int i=0;i<oprow;i++)
			for(int j=0;j<opcol;j++){

				w.set(i, j, Math.random()-0.5);
				//w.set(i, j, 0.35);
			}	
		
		int opbrow = 1, opbcol = 1;		// Since 1 output neuron
		Matrix b = new Matrix(opbrow,opbcol);
		for(int i=0;i<opbrow;i++)
			for(int j=0;j<opbcol;j++){

				b.set(i, j, Math.random()-0.5);
				//b.set(i, j, 0.1);
			}	
		
		// Input and expected output
		int xrow = 4, xcol = 2;
		int[][] iarr = {
				{1,1},
				{1,-1},
				{-1,1},
				{-1,-1}
		};
		Matrix xi = new Matrix(xrow,xcol);
		for(int i=0;i<xrow;i++)
			for(int j=0;j<xcol;j++)
				xi.set(i, j, iarr[i][j]);
		
		int trow = 4, tcol = 1;
		int[][] tarr = {
				{-1},
				{1},
				{1},
				{-1}
		};
		Matrix t = new Matrix(trow,tcol);
		for(int i=0;i<trow;i++)
			for(int j=0;j<tcol;j++)
				t.set(i, j, tarr[i][j]);
		
		// Net input and Output for Hidden Layer
		int nrow = 2, ncol = 1;
		Matrix neth;
		Matrix oph = new Matrix(nrow,ncol);
		
		// Net input and output for Output Neuron
		Matrix net;
		double op = 0;
		
		int k=0;
		double alpha = 1.5005;
		int turns = 100;
		double Etotal=0;
		while(k<turns){
			
			Matrix ip = xi.getMatrix(k%4, k%4, 0, 1);
			neth = hw.times(ip.transpose()).plus(hb);		// Dimension 2*1
			for(int i=0;i<nrow;i++)
				for(int j=0;j<ncol;j++)
					oph.set(i, j, FofX(neth.get(i, j)));	// Dimension 2*1
	
			net = (w.transpose()).times(oph).plus(b);
			op = FofX(net.get(0, 0));		// Fixed since only 1 output neuron

			for(int i=0;i<opbrow;i++){
				
				for(int j=0;j<opbcol;j++){
					
					Etotal+=Math.pow((t.get(k%4, 0)-op),2);
				}
			}
			Etotal/=2;
			
			//Adding error to the Weights and Bias of hidden layer
			for(int i=0;i<hwrow;i++){
				
				for(int j=0;j<hwcol;j++){
					
					double temp = (op-t.get(k%4, 0))*dFofX(net.get(0, 0))*w.get(j,0);
					temp = temp*dFofX(neth.get(j, 0))*ip.get(0, i);
					hw.set(i, j, hw.get(i, j)-alpha*temp);
				}
			}
			
			for(int i=0;i<hbrow;i++){
				
				for(int j=0;j<hbcol;j++){
					
					double temp = (op-t.get(k%4, 0))*dFofX(net.get(0, 0))*w.get(j,0);
					// Since weight for Bias is always 1
					temp = temp*dFofX(neth.get(j, 0))*1;
					//double temp = (op-t.get(k%4, 0))*dFofX(net.get(0, 0));
					hb.set(i, j, hb.get(i, j)-alpha*temp);
				}
			}
			
			double[] dEtotali = new double[nrow];
			for(int i=0;i<nrow;i++){
				
				dEtotali[i] = (op-t.get(k%4, 0))*dFofX(net.get(0, 0))*oph.get(i, 0);
				w.set(i, 0, w.get(i, 0)-alpha*dEtotali[i]);
			}
			for(int i=0;i<opbrow;i++){
				
				for(int j=0;j<opbcol;j++){
					
					// Weight for bias is always 1
					double temp = (op-t.get(k%4, 0))*dFofX(net.get(0, 0));
					b.set(i, 0, b.get(i, 0)-alpha*temp);
				}
			}
			
			System.out.println("Round: "+(k+1)+"\nE: "+Etotal+" O: "+op);
			k++;
			Etotal = 0;
		}
		System.out.println("---Hidden Layer Weights---");
		printMatrix(hw,hwrow,hwcol);
		System.out.println("---Hidden Layer Bias---");
		printMatrix(hb,hbrow,hbcol);
		System.out.println("---Output Layer Weights---");
		printMatrix(w,oprow,opcol);
		System.out.println("---Output Layer Bias---");
		printMatrix(b,opbrow,opbcol);
		//System.out.println(dEtotali[0]*2+" - "+w.get(0, 0));
	}

}
