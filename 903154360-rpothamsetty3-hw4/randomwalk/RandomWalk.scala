package edu.gatech.cse8803.randomwalk

import edu.gatech.cse8803.model.{PatientProperty, EdgeProperty, VertexProperty}
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import scala.collection.mutable._
object RandomWalk {

  def randomWalkOneVsAll(graph: Graph[VertexProperty, EdgeProperty], patientID: Long, numIter: Int = 100, alpha: Double = 0.15): List[Long] = {
    /**
      * Given a patient ID, compute the random walk probability w.r.t. to all other patients.
      * Return a List of patient IDs ordered by the highest to the lowest similarity.
      * For ties, random order is okay
      */

    val sc = graph.vertices.sparkContext
    // broadcast
    val scCasePatients = sc.broadcast(patientID)
    val patientidlist = graph.vertices.filter(f => f._2.isInstanceOf[PatientProperty])
                                      .map(f=> f._1)
                                      .collect().toSet

    val pagerankGraph: Graph[(Boolean, Boolean, Long, Double), Double] = graph
      .outerJoinVertices(graph.outDegrees) { (vid, vdata, deg) => deg.getOrElse(0) }
      .mapTriplets( e => 1.0 / e.srcAttr )
      .mapVertices( (id, attr) =>  if (id.toLong == scCasePatients.value) (true, true, id.toLong, alpha)
                                   else if (patientidlist.contains(id)) (false, true, id.toLong, 0.0)
                                   else (false, false, id.toLong, 0.0)).cache()

    def vertexProgram(id: VertexId, value: (Boolean, Boolean, Long, Double), msgSum: Double): (Boolean, Boolean, Long, Double) =
      if (value._1 && value._2) (true, true, value._3, alpha + (1.0 - alpha) * msgSum)
      else  (false, value._2, value._3, (1.0 - alpha) * msgSum)

    def sendMessage(triplet: EdgeTriplet[(Boolean, Boolean, Long, Double), Double]): Iterator[(VertexId, Double)] = {
      val alpha_check = scala.util.Random.nextFloat > alpha
      if (triplet.srcAttr._1 && alpha_check) Iterator((triplet.dstId, triplet.srcAttr._4*triplet.attr))
      else Iterator.empty
    }
    def messageCombiner(a: Double, b: Double): Double = a + b

    val initialMessage = 0.0

    val pregel_run = Pregel(pagerankGraph, initialMessage, numIter, activeDirection = EdgeDirection.Out)(
      vertexProgram, sendMessage, messageCombiner)
    val similarity_score = pregel_run.vertices.filter(f => f._1.toLong != scCasePatients.value).filter(f => f._2._2).map(f => (f._2._3, f._2._4))
    val ordered = similarity_score.takeOrdered(10)(Ordering[Double].reverse.on(f => f._2)).map(f => f._1).toList
    ordered
  }
}
