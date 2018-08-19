/**

students: please put your implementation in this file!
  **/
package edu.gatech.cse8803.jaccard

import edu.gatech.cse8803.model._
import edu.gatech.cse8803.model.{EdgeProperty, VertexProperty}
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD

object Jaccard {

  def jaccardSimilarityOneVsAll(graph: Graph[VertexProperty, EdgeProperty], patientID: Long): List[Long] = {
    /** 
    Given a patient ID, compute the Jaccard similarity w.r.t. to all other patients. 
    Return a List of top 10 patient IDs ordered by the highest to the lowest similarity.
    For ties, random order is okay. The given patientID should be excluded from the result.
    */

    /** Remove this placeholder and implement your code */

    val direction: EdgeDirection = EdgeDirection.Out
    val neighborIds = graph.collectNeighborIds(direction)
    val patientNeighbors = neighborIds.lookup(patientID).head.toSet
    val patientidlist = graph.vertices.filter(f => f._2.isInstanceOf[PatientProperty])
                                          .map(f=> f._1)
                                          .collect().toSet
    val otherpatients = neighborIds.filter(f=> f._1.toLong != patientID & patientidlist.contains(f._1.toLong))
    val jaccardSimilarityScore = otherpatients.map(f => (f._1, jaccard(patientNeighbors, f._2.toSet)))
                                             .takeOrdered(10)(Ordering[Double].reverse.on(f=> f._2))
                                             .map(f=> f._1).toList

    jaccardSimilarityScore
    }

  def jaccardSimilarityAllPatients(graph: Graph[VertexProperty, EdgeProperty]): RDD[(Long, Long, Double)] = {
    /**
    Given a patient, med, diag, lab graph, calculate pairwise similarity between all
    patients. Return a RDD of (patient-1-id, patient-2-id, similarity) where 
    patient-1-id < patient-2-id to avoid duplications
    */

    /** Remove this placeholder and implement your code */
    val direction: EdgeDirection = EdgeDirection.Out
    val neighborIds = graph.collectNeighborIds(direction)
    val patientidlist = graph.vertices.filter(f => f._2.isInstanceOf[PatientProperty])
                             .map(f=> f._1)
                             .collect().toSet
    val allpatients = neighborIds.filter(f => patientidlist.contains(f._1.toLong))

    /*https://stackoverflow.com/questions/26557873/spark-produce-rddx-x-of-all-possible-combinations-from-rddx*/
    val combination_patients = allpatients.cartesian(allpatients).filter(f => f._1._1 < f._2._1)
    val score_map = combination_patients.map(f => (f._1._1, f._2._1, jaccard(f._1._2.toSet, f._2._2.toSet)))
    score_map
  }

  def jaccard[A](a: Set[A], b: Set[A]): Double = {
    /** 
    Helper function

    Given two sets, compute its Jaccard similarity and return its result.
    If the union part is zero, then return 0.
    */
    
    /** Remove this placeholder and implement your code */
    if (a.union(b).isEmpty){
      return 0.
    }
    val jscore = a.intersect(b).size.toDouble/a.union(b).size
    jscore
  }
}
