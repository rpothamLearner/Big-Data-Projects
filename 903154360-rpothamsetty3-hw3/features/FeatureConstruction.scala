/**
  * @author Hang Su
  */
package edu.gatech.cse8803.features

import edu.gatech.cse8803.model.{LabResult, Medication, Diagnostic}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._


object FeatureConstruction {

  /**
    * ((patient-id, feature-name), feature-value)
    */
  type FeatureTuple = ((String, String), Double)

  /**
    * Aggregate feature tuples from diagnostic with COUNT aggregation,
    * @param diagnostic RDD of diagnostic
    * @return RDD of feature tuples
    */
  def constructDiagnosticFeatureTuple(diagnostic: RDD[Diagnostic]): RDD[FeatureTuple] = {
    /**
      * TODO implement your own code here and remove existing
      * placeholder code
      */
    val diagnostic_ftuple = diagnostic.map(item => ((item.patientID, item.code.toLowerCase()), 1.0)).reduceByKey(_+_)
    diagnostic_ftuple
  }

  /**
    * Aggregate feature tuples from medication with COUNT aggregation,
    * @param medication RDD of medication
    * @return RDD of feature tuples
    */
  def constructMedicationFeatureTuple(medication: RDD[Medication]): RDD[FeatureTuple] = {
    /**
      * TODO implement your own code here and remove existing
      * placeholder code
      */
    val medication_ftuple = medication.map(item => ((item.patientID, item.medicine.toLowerCase()), 1.0)).reduceByKey(_+_)
    medication_ftuple
  }

  /**
    * Aggregate feature tuples from lab result, using AVERAGE aggregation
    * @param labResult RDD of lab result
    * @return RDD of feature tuples
    */
  def constructLabFeatureTuple(labResult: RDD[LabResult]): RDD[FeatureTuple] = {
    /**
      * TODO implement your own code here and remove existing
      * placeholder code
      */
    val labResult_new = labResult.map(item => ((item.patientID, item.testName.toLowerCase()), item.value))
      .groupByKey()
      .map { case ((patientid, testname), value) =>
        var featureSum = value.sum
        var featureCount = value.size.asInstanceOf[Double]
        var featureValue = featureSum/featureCount
        ((patientid, testname), featureValue)
      }
    labResult_new
  }

  /**
    * Aggregate feature tuple from diagnostics with COUNT aggregation, but use code that is
    * available in the given set only and drop all others.
    * @param diagnostic RDD of diagnostics
    * @param candidateCode set of candidate code, filter diagnostics based on this set
    * @return RDD of feature tuples
    */
  def constructDiagnosticFeatureTuple(diagnostic: RDD[Diagnostic], candidateCode: Set[String]): RDD[FeatureTuple] = {
    /**
      * TODO implement your own code here and remove existing
      * placeholder code
      */
    val diagnostic_ftuple = diagnostic.filter(item => candidateCode.contains(item.code.toLowerCase()))
      .map(item => ((item.patientID, item.code), 1.0)).reduceByKey(_+_)
    diagnostic_ftuple
  }

  /**
    * Aggregate feature tuples from medication with COUNT aggregation, use medications from
    * given set only and drop all others.
    * @param medication RDD of diagnostics
    * @param candidateMedication set of candidate medication
    * @return RDD of feature tuples
    */
  def constructMedicationFeatureTuple(medication: RDD[Medication], candidateMedication: Set[String]): RDD[FeatureTuple] = {
    /**
      * TODO implement your own code here and remove existing
      * placeholder code
      */
    val medication_ftuple = medication.filter(item => candidateMedication.contains(item.medicine.toLowerCase()))
      .map(item => ((item.patientID, item.medicine), 1.0)).reduceByKey(_+_)
    medication_ftuple
  }


  /**
    * Aggregate feature tuples from lab result with AVERAGE aggregation, use lab from
    * given set of lab test names only and drop all others.
    * @param labResult RDD of lab result
    * @param candidateLab set of candidate lab test name
    * @return RDD of feature tuples
    */
  def constructLabFeatureTuple(labResult: RDD[LabResult], candidateLab: Set[String]): RDD[FeatureTuple] = {
    /**
      * TODO implement your own code here and remove existing
      * placeholder code
      */
    val labResult_new = labResult.filter(item => candidateLab.contains(item.testName.toLowerCase()))
      .map(item => ((item.patientID, item.testName.toLowerCase()), item.value))
      .groupByKey()
      .map { case ((patientid, testname), value) =>
        var featureSum = value.sum
        var featureCount = value.size.asInstanceOf[Double]
        var featureValue = featureSum / featureCount
        ((patientid, testname), featureValue)
      }
    labResult_new
  }


  /**
    * Given a feature tuples RDD, construct features in vector
    * format for each patient. feature name should be mapped
    * to some index and convert to sparse feature format.
    * @param sc SparkContext to run
    * @param feature RDD of input feature tuples
    * @return
    */
  def construct(sc: SparkContext, feature: RDD[FeatureTuple]): RDD[(String, Vector)] = {

    /** save for later usage */
    feature.cache()

    /** create a feature name to id map*/

    /** transform input feature */

    /**
      * Functions maybe helpful:
      *    collect
      *    groupByKey
      */

    /**
      * TODO implement your own code here and remove existing
      * placeholder code
      */
    feature.cache()
    val featureMap = feature.map(f => f._1._2).distinct.collect.zipWithIndex.toMap
    val scFeatureMap = sc.broadcast(featureMap)
    val numFeature = scFeatureMap.value.size

    def toVector(feature_iter: (String, Iterable[(String, Double)])) : (String, Vector) = {
      val temp = feature_iter._2.map(item => (scFeatureMap.value(item._1), item._2)).toList
      val (x,y) = temp.unzip
      val z: Array[Double] = new Array[Double](numFeature)
      for (i <- 0 to numFeature-1) {
        if(!x.contains(i)) {
          z(i) = 0.0
        }else z(i) = y(x.indexOf(i))
      }
      /*val vec_sparse = Vectors.sparse(numFeature, temp)*/
      val vec_dense = Vectors.dense(z)
      (feature_iter._1, vec_dense)
    }

    val labeledPoint = feature.map(x => (x._1._1, (x._1._2, x._2))).groupByKey().map(toVector)
    labeledPoint
  }

}
