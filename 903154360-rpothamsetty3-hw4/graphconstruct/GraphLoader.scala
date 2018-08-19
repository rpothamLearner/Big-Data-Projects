
/**
  * @author Hang Su <hangsu@gatech.edu>.
  */

package edu.gatech.cse8803.graphconstruct

import edu.gatech.cse8803.model._
import org.apache.spark.SparkContext
import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._


object GraphLoader {
  /** Generate Bipartite Graph using RDDs
    *
    * @input: RDDs for Patient, LabResult, Medication, and Diagnostic
    * @return: Constructed Graph
    *
    * */
  def load(patients: RDD[PatientProperty], labResults: RDD[LabResult],
           medications: RDD[Medication], diagnostics: RDD[Diagnostic]): Graph[VertexProperty, EdgeProperty] = {

    /** HINT: See Example of Making Patient Vertices Below */
        val vertexPatient: RDD[(VertexId, VertexProperty)] = patients
              .map(patient => (patient.patientID.toLong, patient.asInstanceOf[VertexProperty]))

        val lab_offset = patients.map(_.patientID).distinct.count() + 1
        val labVertexIdRDD = labResults.map(_.labName).distinct.zipWithIndex()
                                        .map{ case(medicine, index) =>
                                              (medicine, index + lab_offset) }

        val labVertex = labVertexIdRDD.map{ case(labname, index) => (index, LabResultProperty(labname))}
                           .asInstanceOf[RDD[(VertexId, VertexProperty)]]
        /*
        println(":::::::::::::::::::::::::::::::::::")
        labVertex.take(3).foreach(println)
        */

        val med_offset = labVertexIdRDD.collect.toMap.size + lab_offset + 1
        val medicationVertexIdRDD = medications.map(_.medicine).distinct.zipWithIndex
          .map{ case(medicine, index) =>
            (medicine, index + med_offset) }

        val medVertex = medicationVertexIdRDD.map{ case(medicine, index) => (index, MedicationProperty(medicine))}
          .asInstanceOf[RDD[(VertexId, VertexProperty)]]

        val diagnostic_offset = medicationVertexIdRDD.collect.toMap.size + med_offset + 1
        val diagVertexIdRDD = diagnostics.map(_.icd9code).distinct.zipWithIndex
          .map{ case(code, index) =>
            (code, index + diagnostic_offset) }

        val diagVertex = diagVertexIdRDD.map{ case(code, index) => (index, DiagnosticProperty(code))}
          .asInstanceOf[RDD[(VertexId, VertexProperty)]]


        val sc = patients.sparkContext
        val bcDiag2Vertex = sc.broadcast(diagVertexIdRDD.collect.toMap)
        /* val bcPatient2Vertex = sc.broadcast(patients.map(_.patientID).distinct.collect.toMap) */
        val bcLab2Vertex = sc.broadcast(labVertexIdRDD.collect.toMap)
        val bcMed2Vertex = sc.broadcast(medicationVertexIdRDD.collect.toMap)

        /* : RDD[Edge[EdgeProperty]] */

        def reduce_func_l(a: ((String, String), Iterable[(LabResult)])): LabResult = {
          val recent_date = a._2.map(f => f.date).max
          val op = a._2.filter(f => f.date == recent_date).last
          op
        }

        def reduce_func_d(a: ((String, String), Iterable[(Diagnostic)])): Diagnostic = {
          val recent_date = a._2.map(f => f.date).max
          val op = a._2.filter(f => f.date == recent_date).last
          op
        }

        def reduce_func_m(a:((String, String), Iterable[(Medication)])): Medication = {
          val recent_date = a._2.map(f => f.date).max
          val op = a._2.filter(f => f.date == recent_date).last
          op
        }

        val edgeLab2Patient: RDD[Edge[EdgeProperty]] = labResults.map(event => ((event.patientID, event.labName), event))
          .groupByKey().map(reduce_func_l)
          .map(f => ((f.patientID, f.labName), f))
          .map{case((patientId, labName), labresult) =>
            Edge( patientId.toLong, // src id
              bcLab2Vertex.value(labName), // target id
              PatientLabEdgeProperty(labresult) // edge property
            )}

        val edgeDiagnostic2Patient: RDD[Edge[EdgeProperty]] = diagnostics.map(event => ((event.patientID, event.icd9code), event))
          .groupByKey().map(reduce_func_d)
          .map(f => ((f.patientID, f.icd9code), f))
          .map{case((patientId, icd9), diagnostic) =>
            Edge( patientId.toLong, // src id
              bcDiag2Vertex.value(icd9), // target id
              PatientDiagnosticEdgeProperty(diagnostic) // edge property
            )}
/*
        println("error here")
        edgeDiagnostic2Patient.take(3).foreach(println) */

        val edgeMed2Patient: RDD[Edge[EdgeProperty]] = medications.map(event => ((event.patientID, event.medicine), event))
                .groupByKey().map(reduce_func_m)
                .map(f => ((f.patientID, f.medicine), f))
                .map{case((patientId, medicine), medication) =>
                  Edge( patientId.toLong, // src id
                    bcMed2Vertex.value(medicine), // target id
                    PatientMedicationEdgeProperty(medication) // edge property
                  )}

        val vertices = sc.union(vertexPatient, diagVertex, labVertex, medVertex)
        val edges = sc.union(edgeLab2Patient, edgeDiagnostic2Patient, edgeMed2Patient)

    /*
    diagVertex.take(3).foreach(println)
    vertexPatient.take(3).foreach(println)
    vertices.take(3).foreach(println)

    //Making Graph
    print("---------------")
    println(edges.take(2).foreach(println))
    print("---------------")
    println(vertices.take(2).foreach(println)) */
        val graph: Graph[VertexProperty, EdgeProperty] = Graph(vertices, edges)
        val graph_rev = graph.reverse
        val bidirectional_graph = Graph(vertices, graph.edges.union(graph_rev.edges))
   /* println("graph_stats::::::::::")
    println(bidirectional_graph.numEdges)
    println(bidirectional_graph.numVertices)
    bidirectional_graph.vertices.foreach(println) */
    bidirectional_graph
  }
}
