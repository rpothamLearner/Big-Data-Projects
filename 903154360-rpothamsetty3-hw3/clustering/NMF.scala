package edu.gatech.cse8803.clustering

/**
  * @author Hang Su <hangsu@gatech.edu>
  */

import breeze.linalg.{sum, DenseMatrix => BDM, DenseVector => BDV}
import breeze.linalg._
import breeze.numerics._
import org.apache.spark.mllib.linalg.{Matrices, Vector, Vectors}
import org.apache.spark.mllib.linalg.distributed.RowMatrix


object NMF {

  /**
   * Run NMF clustering 
   * @param V The original non-negative matrix 
   * @param k The number of clusters to be formed, also the number of cols in W and number of rows in H
   * @param maxIterations The maximum number of iterations to perform
   * @param convergenceTol The maximum change in error at which convergence occurs.
   * @return two matrixes W and H in RowMatrix and DenseMatrix format respectively 
   */
  def run(V: RowMatrix, k: Int, maxIterations: Int, convergenceTol: Double = 1e-4): (RowMatrix, BDM[Double]) = {

    /**
     * TODO 1: Implement your code here
     * Initialize W, H randomly 
     * Calculate the initial error (Euclidean distance between V and W * H)
     */

    var W = new RowMatrix(V.rows.map(_ => BDV.rand[Double](k)).map(fromBreeze))

    var H = BDM.rand[Double](k, V.numCols().toInt)

    def calculateError(V: RowMatrix, W: RowMatrix, H: DenseMatrix[Double]): Double ={
      var WH = multiply(W, H)
      val rows = V.rows.zip(WH.rows)
                       .map{ case (v1: Vector, v2: Vector) => toBreezeVector(v1) :- toBreezeVector(v2) }
                       .map(fromBreeze)
      val V_WH = new RowMatrix(rows)
      val squared_err = dotProd(V_WH, V_WH).rows.map(item => item.toArray).map(f => f.sum).sum()
      squared_err*0.5
    }

    var error_prev = 0.0
    var err_diff = 0.0
    var error_cur = 0.0
    /**
     * TODO 2: Implement your code here
     * Iteratively update W, H in a parallel fashion until error falls below the tolerance value 
     * The updating equations are, 
     * H = H.* W^T^V ./ (W^T^W H)
     * W = W.* VH^T^ ./ (W H H^T^)
     */
    var iteration_cur = 1
    do {
      error_prev = calculateError(V, W, H)
      var HHt = H*H.t
      var VHt = multiply(V, H.t)
      var WHHt = multiply(W, HHt)
      W =  dotProd(W, dotDiv(VHt, WHHt))
      var WtV = computeWTV(W, V)
      var WtW = computeWTV(W, W)
      H = H:*(WtV :/ (WtW*H :+ 1.e-10))
       /*zero value div taken care in dotDiv function */

      error_cur = calculateError(V, W, H)
      err_diff = abs(error_cur -  error_prev)
      iteration_cur += 1

      W.rows.cache()
      V.rows.cache()
      println(f"iteration number $iteration_cur and error is $error_cur")
    } while (iteration_cur < maxIterations && err_diff > convergenceTol)

    (W, H)
  }


  /**  
  * RECOMMENDED: Implement the helper functions if you needed
  * Below are recommended helper functions for matrix manipulation
  * For the implementation of the first three helper functions (with a null return), 
  * you can refer to dotProd and dotDiv whose implementation are provided
  */
  /**
  * Note:You can find some helper functions to convert vectors and matrices
  * from breeze library to mllib library and vice versa in package.scala
  */

  /** compute the mutiplication of a RowMatrix and a dense matrix */
  def multiply(X: RowMatrix, d: BDM[Double]): RowMatrix = {
    X.multiply(fromBreeze(d))
  }

 /** get the dense matrix representation for a RowMatrix */
  def getDenseMatrix(X: RowMatrix): BDM[Double] = {

    /* other method */
    val array = X.rows.map(f => f.toArray).collect()
    val dm = DenseMatrix(array.map(_.toArray):_*)
    val nrows = X.numRows()
    val ncols = X.numCols()
    val x = X.rows.map(f => f.toArray).collect().flatten

    val denseM: DenseMatrix[Double] = new DenseMatrix(nrows.toInt, ncols.toInt, x)
    dm
  }

  /** matrix multiplication of W.t and V */
  def computeWTV(W: RowMatrix, V: RowMatrix): BDM[Double] = {
    val w_dense = getDenseMatrix(W)
    val v_dense = getDenseMatrix(V)
    w_dense.t*v_dense
  }

  /** dot product of two RowMatrixes */
  def dotProd(X: RowMatrix, Y: RowMatrix): RowMatrix = {
    val rows = X.rows.zip(Y.rows).map{case (v1: Vector, v2: Vector) =>
      toBreezeVector(v1) :* toBreezeVector(v2)
    }.map(fromBreeze)
    new RowMatrix(rows)
  }

  /** dot division of two RowMatrixes */
  def dotDiv(X: RowMatrix, Y: RowMatrix): RowMatrix = {
    val rows = X.rows.zip(Y.rows).map{case (v1: Vector, v2: Vector) =>
      toBreezeVector(v1) :/ toBreezeVector(v2).mapValues(_ + 2.0e-15)
    }.map(fromBreeze)
    new RowMatrix(rows)
  }

}
