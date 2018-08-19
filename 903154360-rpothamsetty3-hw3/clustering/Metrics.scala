/**
  * @author Hang Su <hangsu@gatech.edu>.
  */

package edu.gatech.cse8803.clustering

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.rdd.RDD

object Metrics {
  /**
    * Given input RDD with tuples of assigned cluster id by clustering,
    * and corresponding real class. Calculate the purity of clustering.
    * Purity is defined as
    *             \fract{1}{N}\sum_K max_j |w_k \cap c_j|
    * where N is the number of samples, K is number of clusters and j
    * is index of class. w_k denotes the set of samples in k-th cluster
    * and c_j denotes set of samples of class j.
    * @param clusterAssignmentAndLabel RDD in the tuple format
    *                                  (assigned_cluster_id, class)
    * @return purity
    */
  def purity(clusterAssignmentAndLabel: RDD[(Int, Int)]): Double = {
    /**
      * TODO: Remove the placeholder and implement your code here
      */
    clusterAssignmentAndLabel.cache()
    val N = clusterAssignmentAndLabel.count()

    def count_cluster(iter: (Int, Iterable[(Int, Int)])): Double =  {
      val temp = iter._2.groupBy(f => f._2)
        .map(f=> f._2.map(x => 1.0).sum)
        .reduce((a,b) => a max b)
      temp
    }

    val cluster_grp1 = clusterAssignmentAndLabel.map(item => (item._1, (item._1, item._2))).groupByKey().map(count_cluster).sum()


    val purity = cluster_grp1/N.asInstanceOf[Double]
    purity
  }

}
