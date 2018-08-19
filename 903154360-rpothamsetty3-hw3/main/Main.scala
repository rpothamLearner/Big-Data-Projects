/**
  * @author Hang Su <hangsu@gatech.edu>.
  */

package edu.gatech.cse8803.main

import java.text.SimpleDateFormat

import edu.gatech.cse8803.clustering.{NMF, Metrics}
import edu.gatech.cse8803.features.FeatureConstruction
import edu.gatech.cse8803.ioutils.CSVUtils
import edu.gatech.cse8803.model.{Diagnostic, LabResult, Medication}
import edu.gatech.cse8803.phenotyping.T2dmPhenotype
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.clustering.{GaussianMixture, KMeans, StreamingKMeans}
import org.apache.spark.mllib.linalg.{DenseMatrix, Matrices, Vectors, Vector}
import org.apache.spark.mllib.feature.StandardScaler
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.storage.StorageLevel

import scala.io.Source


object Main {
  def main(args: Array[String]) {
    import org.apache.log4j.Logger
    import org.apache.log4j.Level

    Logger.getLogger("org").setLevel(Level.WARN)
    Logger.getLogger("akka").setLevel(Level.WARN)

    val sc = createContext
    val sqlContext = new SQLContext(sc)

    /** initialize loading of data */
    val (medication, labResult, diagnostic) = loadRddRawData(sqlContext)
    val (candidateMedication, candidateLab, candidateDiagnostic) = loadLocalRawData

    /** conduct phenotyping */
    val phenotypeLabel = T2dmPhenotype.transform(medication, labResult, diagnostic).cache()

    /** feature construction with all features */
    val featureTuples = sc.union(
      FeatureConstruction.constructDiagnosticFeatureTuple(diagnostic),
      FeatureConstruction.constructLabFeatureTuple(labResult),
      FeatureConstruction.constructMedicationFeatureTuple(medication)
    )

    val rawFeatures = FeatureConstruction.construct(sc, featureTuples)

    val (kMeansPurity, gaussianMixturePurity, streamKmeansPurity, nmfPurity) = testClustering(phenotypeLabel, rawFeatures)
    println(f"[All feature] purity of kMeans is: $kMeansPurity%.5f")
    println(f"[All feature] purity of GMM is: $gaussianMixturePurity%.5f")
    println(f"[All feature] purity of StreamingKMeans is: $streamKmeansPurity%.5f")
    println(f"[All feature] purity of NMF is: $nmfPurity%.5f")

    /** feature construction with filtered features */
    val filteredFeatureTuples = sc.union(
      FeatureConstruction.constructDiagnosticFeatureTuple(diagnostic, candidateDiagnostic),
      FeatureConstruction.constructLabFeatureTuple(labResult, candidateLab),
      FeatureConstruction.constructMedicationFeatureTuple(medication, candidateMedication)
    )

    val filteredRawFeatures = FeatureConstruction.construct(sc, filteredFeatureTuples)

    val (kMeansPurity2, gaussianMixturePurity2, streamKmeansPurity2, nmfPurity2) = testClustering(phenotypeLabel, filteredRawFeatures)
    println(f"[Filtered feature] purity of kMeans is: $kMeansPurity2%.5f")
    println(f"[Filtered feature] purity of GMM is: $gaussianMixturePurity2%.5f")
    println(f"[Filtered feature] purity of StreamingKMeans is: $streamKmeansPurity2%.5f")
    println(f"[Filtered feature] purity of NMF is: $nmfPurity2%.5f")
    sc.stop
  }

  def testClustering(phenotypeLabel: RDD[(String, Int)], rawFeatures:RDD[(String, Vector)]): (Double, Double, Double, Double) = {
    import org.apache.spark.mllib.linalg.Matrix
    import org.apache.spark.mllib.linalg.distributed.RowMatrix
    phenotypeLabel.cache()
    /** scale features */
    val scaler = new StandardScaler(withMean = true, withStd = true).fit(rawFeatures.map(_._2))
    val features = rawFeatures.map({ case (patientID, featureVector) => (patientID, scaler.transform(Vectors.dense(featureVector.toArray)))})
    val rawFeatureVectors = features.map(_._2).cache()

    /** reduce dimension */
    val mat: RowMatrix = new RowMatrix(rawFeatureVectors)
    val pc: Matrix = mat.computePrincipalComponents(10) // Principal components are stored in a local dense matrix.
    val featureVectors = mat.multiply(pc).rows.cache()

    val densePc = Matrices.dense(pc.numRows, pc.numCols, pc.toArray).asInstanceOf[DenseMatrix]
    /** transform a feature into its reduced dimension representation */
    def transform(feature: Vector): Vector = {
      Vectors.dense(Matrices.dense(1, feature.size, feature.toArray).multiply(densePc).toArray)
    }

    /** TODO: K Means Clustering using spark mllib
      *  Train a k means model using the variabe featureVectors as input
      *  Set maxIterations =20 and seed as 8803L
      *  Assign each feature vector to a cluster(predicted Class)
      *  Obtain an RDD[(Int, Int)] of the form (cluster number, RealClass)
      *  Find Purity using that RDD as an input to Metrics.purity
      *  Remove the placeholder below after your implementation
      **/
    /* */
    val numClusters = 3
    val numIterations = 20
    val seed = 8803L

    val kmeans = new KMeans().setK(numClusters).setMaxIterations(numIterations).setSeed(seed).run(featureVectors)
    val kmeans_cluster = kmeans.predict(featureVectors)
    val clusterWithPatientIds = features.map({case (patientId,f)=>patientId}).zip(kmeans_cluster)
    val kmeansClusterAssignmentAndLabel = clusterWithPatientIds.join(phenotypeLabel).map({case (patientID,value)=>value}).cache()

    /*.zipWithIndex().map(item => (item._2, item._1)) */


    val kMeansPurity = Metrics.purity(kmeansClusterAssignmentAndLabel)


    /** TODO: GMMM Clustering using spark mllib
      *  Train a Gaussian Mixture model using the variabe featureVectors as input
      *  Set maxIterations =20 and seed as 8803L
      *  Assign each feature vector to a cluster(predicted Class)
      *  Obtain an RDD[(Int, Int)] of the form (cluster number, RealClass)
      *  Find Purity using that RDD as an input to Metrics.purity
      *  Remove the placeholder below after your implementation
      **/
    /*
      val gmm = new GaussianMixture().setK(numClusters).setMaxIterations(numIterations).setSeed(seed).run(featureVectors)
      val gmm_cluster = gmm.predict(featureVectors).zipWithIndex().map(item => (item._2, item._1))
      val rdd_purity1 = gmm_cluster.join(realClass).map(item => (item._2._1, item._2._2)) */

    val gmm = new GaussianMixture().setK(numClusters).setMaxIterations(numIterations).setSeed(seed).run(featureVectors)
    val gmm_cluster = gmm.predict(featureVectors)
    val ModelsWithPatientIds = features.map({case (patientId,f)=>patientId}).zip(gmm_cluster)
    val gmmClusterAssignmentAndLabel = ModelsWithPatientIds.join(phenotypeLabel).map({case (patientID,value)=>value}).cache()
    val gaussianMixturePurity = Metrics.purity(gmmClusterAssignmentAndLabel)


    /** TODO: StreamingKMeans Clustering using spark mllib
      *  Train a StreamingKMeans model using the variabe featureVectors as input
      *  Set the number of cluster K = 3 and DecayFactor = 1.0, seed as 8803L and weight as 0.0
      *  please pay attention to the input type
      *  Assign each feature vector to a cluster(predicted Class)
      *  Obtain an RDD[(Int, Int)] of the form (cluster number, RealClass)
      *  Find Purity using that RDD as an input to Metrics.purity
      *  Remove the placeholder below after your implementation
      *  In StreamingKMeans, a model is continuously updated, so you need
      *                   to use the latestModel() on the StreamingKMeans object.
      **/

    val streamK = new StreamingKMeans().setK(numClusters)
                        .setDecayFactor(1.0)
                        .setRandomCenters(10,0.0, 8803L)
                        .latestModel()
                        .update(featureVectors, 1.0, "batches")
                        .predict(featureVectors)
    val StreamKWithPatientIds = features.map({case (patientId,f)=>patientId}).zip(streamK)
    val StreamKAssignmentAndLabel = StreamKWithPatientIds.join(phenotypeLabel).map({case (patientID,value)=>value})
    val streamKmeansPurity = Metrics.purity(StreamKAssignmentAndLabel)


   /*  */
    /** NMF  */
    val rawFeaturesNonnegative = rawFeatures.map({ case (patientID, f)=> Vectors.dense(f.toArray.map(v=>Math.abs(v)))})
    val (w, _) = NMF.run(new RowMatrix(rawFeaturesNonnegative), numClusters, 100)
    // for each row (patient) in W matrix, the index with the max value should be assigned as its cluster type
    val assignments = w.rows.map(_.toArray.zipWithIndex.maxBy(_._1)._2)
    // zip patientIDs with their corresponding cluster assignments
    // Note that map doesn't change the order of rows

    val assignmentsWithPatientIds=features.map({case (patientId,f)=>patientId}).zip(assignments)
    // join your cluster assignments and phenotypeLabel on the patientID and obtain a RDD[(Int,Int)]
    // which is a RDD of (clusterNumber, phenotypeLabel) pairs
    val nmfClusterAssignmentAndLabel = assignmentsWithPatientIds.join(phenotypeLabel).map({case (patientID,value)=>value})
    // Obtain purity value
    val nmfPurity = Metrics.purity(nmfClusterAssignmentAndLabel)


    (kMeansPurity, gaussianMixturePurity, streamKmeansPurity, nmfPurity)
  }

  /**
    * load the sets of string for filtering of medication
    * lab result and diagnostics
    *
    * @return
    */
  def loadLocalRawData: (Set[String], Set[String], Set[String]) = {
    val candidateMedication = Source.fromFile("data/med_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    val candidateLab = Source.fromFile("data/lab_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    val candidateDiagnostic = Source.fromFile("data/icd9_filter.txt").getLines().map(_.toLowerCase).toSet[String]
    (candidateMedication, candidateLab, candidateDiagnostic)
  }

  def loadRddRawData(sqlContext: SQLContext): (RDD[Medication], RDD[LabResult], RDD[Diagnostic]) = {
    /** You may need to use this date format. */
    val dateFormat = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ssX")

    /** load data using Spark SQL into three RDDs and return them
      * Hint: You can utilize edu.gatech.cse8803.ioutils.CSVUtils and SQLContext.
      *
      * Notes:Refer to model/models.scala for the shape of Medication, LabResult, Diagnostic data type.
      *       Be careful when you deal with String and numbers in String type.
      *       Ignore lab results with missing (empty or NaN) values when these are read in.
      *       For dates, use Date_Resulted for labResults and Order_Date for medication.
      * */

    val medication_df  = CSVUtils.loadCSVAsTable(sqlContext, "data/medication_orders_INPUT.csv", "medication")
    val labResult_df = CSVUtils.loadCSVAsTable(sqlContext, "data/lab_results_INPUT.csv", "labResult")
    val diagnostic_df = CSVUtils.loadCSVAsTable(sqlContext, "data/encounter_dx_INPUT.csv", "diagnostic")

    val medication_df1 = CSVUtils.loadCSVAsTable(sqlContext, "data/medication_orders_INPUT.csv", "medication").select("Member_ID", "Order_Date", "Drug_Name")
    val medication_rdd: RDD[Medication] = medication_df1.map(x => Medication(x.getString(0), dateFormat.parse(x.getString(1)), x.getString(2)))

    medication_df1.unpersist()
    medication_df.unpersist()

    val labResult_df_x = CSVUtils.loadCSVAsTable(sqlContext, "data/lab_results_INPUT.csv", "labResult").select("Member_ID", "Date_Resulted", "Result_Name", "Numeric_Result")
    val labResult_df1 = labResult_df_x.withColumnRenamed("Numeric_Result", "Tmp")
    /* labResult_df1.take(5).foreach(println) */
    labResult_df_x.unpersist()
    labResult_df1.unpersist()
    labResult_df.unpersist()

    val labResult_df2 = labResult_df1.withColumn("Numeric_Result", labResult_df1("Tmp").cast("double"))
    val labResult_df3 = labResult_df2.filter(labResult_df2.col("Numeric_Result").isNotNull)
    val labResult_rdd: RDD[LabResult] = labResult_df3.map(x => LabResult(x.getString(0), dateFormat.parse(x.getString(1)), x.getString(2), x.getDouble(4)))
    /*labResult_rdd.take(5).foreach(println)*/

    labResult_df2.unpersist()
    labResult_df3.unpersist()

    /* medication_rdd.saveAsTextFile("/Users/rajeshpothamsetty/Dropbox/Rajesh_DB/IMP_Dropbox/OMSCS/BigData/ttt.tx")
    labResult_rdd.saveAsTextFile("/Users/rajeshpothamsetty/Dropbox/Rajesh_DB/IMP_Dropbox/OMSCS/BigData/rrr.txt") */
    val diagnostic_df1 = CSVUtils.loadCSVAsTable(sqlContext, "data/encounter_INPUT.csv", "diagnostic").select("Encounter_ID", "Member_ID", "Encounter_DateTime")
    val diagnostic_df2 = CSVUtils.loadCSVAsTable(sqlContext, "data/encounter_dx_INPUT.csv", "Diagnostic").select("Encounter_ID", "code")
    val diagnostic_join = diagnostic_df1.join(diagnostic_df2,diagnostic_df1("Encounter_ID") === diagnostic_df2("Encounter_ID"))
    /*diagnostic_join.take(5).foreach(println)*/
    val diagnostic_join1 = diagnostic_join.select("Member_ID", "Encounter_DateTime", "code").na.drop()
    val diagnostic_rdd: RDD[Diagnostic] = diagnostic_join1.map(x => Diagnostic(x.getString(0), dateFormat.parse(x.getString(1)), x.getString(2)))

    diagnostic_df.unpersist()
    diagnostic_df1.unpersist()
    diagnostic_join.unpersist()
    diagnostic_join1.unpersist()

    (medication_rdd, labResult_rdd, diagnostic_rdd)
  }

  def createContext(appName: String, masterUrl: String): SparkContext = {
    val conf = new SparkConf().setAppName(appName).setMaster(masterUrl)
    new SparkContext(conf)
  }

  def createContext(appName: String): SparkContext = createContext(appName, "local")

  def createContext: SparkContext = createContext("CSE 8803 Homework Two Application", "local")
}
