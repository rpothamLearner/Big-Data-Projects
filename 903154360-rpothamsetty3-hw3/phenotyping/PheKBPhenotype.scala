/**
  * @author Hang Su <hangsu@gatech.edu>,
  * @author Sungtae An <stan84@gatech.edu>,
  */

package edu.gatech.cse8803.phenotyping

import edu.gatech.cse8803.model.{Diagnostic, LabResult, Medication}
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.RDD._
import org.apache.spark.SparkContext._

import scala.Tuple2

object T2dmPhenotype {

  // criteria codes given
  val T1DM_DX = Set("250.01", "250.03", "250.11", "250.13", "250.21", "250.23", "250.31", "250.33", "250.41", "250.43",
    "250.51", "250.53", "250.61", "250.63", "250.71", "250.73", "250.81", "250.83", "250.91", "250.93")

  val T2DM_DX = Set("250.3", "250.32", "250.2", "250.22", "250.9", "250.92", "250.8", "250.82", "250.7", "250.72", "250.6",
    "250.62", "250.5", "250.52", "250.4", "250.42", "250.00", "250.02")

  val T1DM_MED = Set("lantus", "insulin glargine", "insulin aspart", "insulin detemir", "insulin lente", "insulin nph", "insulin reg", "insulin,ultralente")

  val T2DM_MED = Set("chlorpropamide", "diabinese", "diabanase", "diabinase", "glipizide", "glucotrol", "glucotrol xl",
    "glucatrol ", "glyburide", "micronase", "glynase", "diabetamide", "diabeta", "glimepiride", "amaryl",
    "repaglinide", "prandin", "nateglinide", "metformin", "rosiglitazone", "pioglitazone", "acarbose",
    "miglitol", "sitagliptin", "exenatide", "tolazamide", "acetohexamide", "troglitazone", "tolbutamide",
    "avandia", "actos", "actos", "glipizide")

  val DM_RELATED_DX = Set("790.21", "790.22", "790.2", "790.29", "648.81", "648.82", "648.83", "648.84", "684", "648.0", "648.00", "648.01", "648.02",
    "648.03", "648.04", "791.5", "277.7", "V77.1", "v77.1","256.4", "250.*", "250", "250.0", "250.00", "250.01")


  /**
    * Transform given data set to a RDD of patients and corresponding phenotype
    * @param medication medication RDD
    * @param labResult lab result RDD
    * @param diagnostic diagnostic code RDD
    * @return tuple in the format of (patient-ID, label). label = 1 if the patient is case, label = 2 if control, 3 otherwise
    */
  def transform(medication: RDD[Medication], labResult: RDD[LabResult], diagnostic: RDD[Diagnostic]): RDD[(String, Int)] = {
    /**
      * Remove the place holder and implement your code here.
      * Hard code the medication, lab, icd code etc. for phenotypes like example code below.
      * When testing your code, we expect your function to have no side effect,
      * i.e. do NOT read from file or write file
      *
      * You don't need to follow the example placeholder code below exactly, but do have the same return type.
      *
      * Hint: Consider case sensitivity when doing string comparisons.
      * https://stackoverflow.com/questions/35569062/how-to-retrieve-record-with-min-value-in-spark
      */


    /** Hard code the criteria */
    // val type1_dm_dx = Set("code1", "250.03")
    // val type1_dm_med = Set("med1", "insulin nph")
    // use the given criteria above like T1DM_DX, T2DM_DX, T1DM_MED, T2DM_MED and hard code DM_RELATED_DX criteria as well

    val casePatients = labelCase(diagnostic, medication).cache()

    val controlPatients = labelControl(diagnostic, labResult).cache()

    val case_plus_control = casePatients.map(f => f._1).union(controlPatients.map(f => f._1))

    val otherPatients = diagnostic.map(f => f.patientID).distinct().subtract(case_plus_control).map(f => (f, 3))

    val phenotypeLabel = casePatients.union(controlPatients).union(otherPatients)

    phenotypeLabel
  }

  def labelCase(diagnostic: RDD[Diagnostic], medication: RDD[Medication]): RDD[(String, Int)] ={
    /*
    cond_1 = Not Type 1 DM diagnosis
    cond_2 = cond_1 + Type 2 DM diagnosis
    cond_3 = cond_2 + Not Type 1 DM medication
    cond_4 = cond_2 + Type 1 DM medication
    cond_5 = cond_4 + Not Type 2 DM medication
    cond_6 = cond_4 + Type 2 DM medication
    cond_7 (t1_t2dm_precedes) = T1MED + T2MED And (T2 < T1)
     */
    diagnostic.cache()
    medication.cache()

    val cond_1 = diagnostic.filter(item => !T1DM_DX.contains(item.code))
    val cond_2 = cond_1.filter(item => T2DM_DX.contains(item.code)).map(f => f.patientID).distinct()

    val T1DM_MED_patientID = medication.filter(item => T1DM_MED.contains(item.medicine.toLowerCase()))
      .map(f => f.patientID).collect().toSet
    val cond_3 = cond_2.filter(item => !T1DM_MED_patientID.contains(item)).cache()

    val cond_4 = cond_2.filter(item => T1DM_MED_patientID.contains(item)).cache()
    val T2DM_MED_patientID = medication.filter(item => T2DM_MED.contains(item.medicine.toLowerCase()))
      .map(f => f.patientID).collect().toSet
    val cond_5 = cond_4.filter(item => !T2DM_MED_patientID.contains(item))
    val cond_6 = cond_4.filter(item => T2DM_MED_patientID.contains(item)).collect().toSet

    val t1dm_precedes: RDD[Medication] = medication.filter(item => T1DM_MED.contains(item.medicine.toLowerCase()))
    val t2dm_precedes: RDD[Medication] = medication.filter(item => T2DM_MED.contains(item.medicine.toLowerCase()))
    /* t1dm_precedes.take(5).foreach(println) */

    val t1dm_min = t1dm_precedes.map(f => (f.patientID, (f.date, f.medicine)))
      .reduceByKey((a, b) => if (a._1.before(b._1)) a else b)
      .map(f => (f._1, (f._2._1, f._2._2)))

    val t2dm_min = t2dm_precedes.map(f => (f.patientID, (f.date, f.medicine)))
      .reduceByKey((a, b) => if (a._1.before(b._1)) a else b)
      .map(f => (f._1, (f._2._1, f._2._2)))

    val t1_t2dm = t1dm_min.join(t2dm_min)
    /*
    println("join")
    var y = t1_t2dm.count()
    println(y)
    t1_t2dm.take(5).foreach(println) */
    val cond_7 = t1_t2dm.filter(item => item._2._2._1.before(item._2._1._1)).map(item => item._1)

    /*val t1_t2db = t1dm_min.join()

    t1dm_min.take(5).foreach(println)
    t2dm_min.take(5).foreach(println)

    println("cond1 - 3002")
    println(cond_1.count())
    cond_2.take(5).foreach(println)
    println("cond_2 - 1265")
    println(cond_2.count())
    cond_3.count()
    println("cond_3 - 427")
    println(cond_3.count())
    cond_4.count()
    println("cond_2 - 838")
    println(cond_4.count())
    cond_5.count()
    println("cond_5 - 255")
    println(cond_5.count())
    println("cond_6 - 583")
    println(cond_6.size)
    println("final_case - (294)")
    var x = t1_t2dm_precedes.count()
    println(x)
    t1_t2dm_precedes.take(5).foreach(println)*/
    println(s"XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

    val all_case = cond_3.union(cond_5).union(cond_7).map(x => (x, 1))
    /*println("all case - 976")
    println(all_case.count())*/
    all_case
  }

  def labelControl(diagnostic: RDD[Diagnostic], labResult: RDD[LabResult]): RDD[(String, Int)] ={

    labResult.cache()

    val glucose_lab = Set( "fasting glucose", "fasting blood glucose", "fasting plasma glucose", "glucose","glucose, serum", "glucose 2hr post dose", "glucose 3hr")

    val glucose_data = labResult.filter(item => glucose_lab.contains(item.testName.toLowerCase())).map(f => f.patientID).distinct()

    /* println("glucose - 1823")
    println(glucose_data.count()) */

    def isAbnormal(result: LabResult): Boolean = {
      var isabnormal = false
      val abnormal_set1 = Set(("hba1c", 6.0), ("hemoglobin a1c", 6.0), ("fasting glucose",110.0), ("fasting blood glucose", 110.0), ("fasting plasma glucose", 110.0))
      val abnormal_set2 = Set(("glucose",110.0), ("glucose, serum", 110.0))

      for (x <- abnormal_set1) {
        /*println(x._1)
        println(x._2)
        println(x._1 == result.testName.toLowerCase)
        println(x._2.asInstanceOf[Double] >= result.value)*/
        if (x._1 == result.testName.toLowerCase && result.value >= x._2.asInstanceOf[Double]) {
          isabnormal = true
          /*println("yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy")
          println(result.testName)*/
        }
      }
      for (x <- abnormal_set2) {
        if (x._1 == result.testName.toLowerCase && result.value > x._2.asInstanceOf[Double]) {
          isabnormal = true
          /*println("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
          println(result.testName)*/
        }
      }
      isabnormal
    }


    /*
     val abnormal = labResult.filter(item => glucose_lab.contains(item.testName.toLowerCase())).filter(x => !isAbnormal(x)).map(item=> item.patientID).map(x => x._1).distinct()
     val gluc_temp = labResult.filter(item => glucose_lab.contains(item.testName.toLowerCase()))
     /*  val temp1 = gluc_temp.map(item=> (item.patientID, (item.date, item.testName, item.value))).leftOuterJoin(temp).map(f => f._1).distinct()
    .filter(f => f._1 == null) */
     println("new abnormal count - ")
     println(temp.count())

     val filtered_data2 = labResult.filter(item => glucose_lab.contains(item.testName.toLowerCase())).filter(x => !isAbnormal(x))
     val filtered_data3 = labResult.filter(item => glucose_lab.contains(item.testName.toLowerCase())).filter(x => isAbnormal(x))
     */

    val filtered_data = labResult.filter(x => isAbnormal(x)).map(f => f.patientID).distinct()
    val gluc_NotAbnormal = glucose_data.subtract(filtered_data)

    /* println("After abnormal - 953")
    println(gluc_NotAbnormal.count()) */
    val diag_dm = diagnostic.filter(f => DM_RELATED_DX.contains(f.code.toLowerCase)).map(f => f.patientID).distinct()
    val filtered_final = gluc_NotAbnormal.subtract(diag_dm)
    /* println("final - 948")
    println(filtered_final.count()) */
    filtered_final.map(f => (f, 2))
  }
}

