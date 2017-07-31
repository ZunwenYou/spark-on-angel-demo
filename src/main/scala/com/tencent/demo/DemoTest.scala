package com.tencent.demo

import scala.collection.mutable.ArrayBuffer

import breeze.linalg.DenseVector
import breeze.optimize.StochasticGradientDescent
import org.apache.spark.SparkConf
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession

import com.tencent.angel.spark.PSContext
import com.tencent.angel.spark.examples.util.Logistic
import com.tencent.angel.spark.models.vector.BreezePSVector

object DemoTest {

  def main(args: Array[String]): Unit = {
    val dim = 100
    val numSample = 200
    val numSlices = 10
    val maxIter = 50

    val conf = new SparkConf
    val master = conf.getOption("spark.master")
    val isLocalTest = if (master.isEmpty || master.get.toLowerCase.startsWith("local")) true else false
    val builder = SparkSession.builder().appName(this.getClass.getSimpleName)
    if (isLocalTest) {
      builder.master("local")
        .config("spark.ps.mode", "LOCAL")
        .config("spark.ps.jars", "")
        .config("spark.ps.instances", "1")
        .config("spark.ps.cores", "1")
    }
    val spark = builder.getOrCreate()

    PSContext.getOrCreate(spark.sparkContext)
    execute(dim, numSample, numSlices, maxIter)
  }

  def execute(
      dim: Int,
      sampleNum: Int,
      partitionNum: Int,
      maxIter: Int, stepSize:
  Double = 0.1): Unit = {

    val trainData = Logistic.generateLRData(sampleNum, dim, partitionNum)

    // runSGD
    var startTime = System.currentTimeMillis()
    runSGD(trainData, dim, stepSize, maxIter)
    var endTime = System.currentTimeMillis()
    println(s"SGD time: ${endTime - startTime}")

    // run PS SGD
    startTime = System.currentTimeMillis()
    runPsSGD(trainData, dim, stepSize, maxIter)
    endTime = System.currentTimeMillis()
    println(s"PS SGD time: ${endTime - startTime} ")

    // run PS aggregate SGD
    startTime = System.currentTimeMillis()
    runPsAggregateSGD(trainData, dim, stepSize, maxIter)
    endTime = System.currentTimeMillis()
    println(s"PS aggregate SGD time: ${endTime - startTime} ")
  }

  def runSGD(trainData: RDD[(Vector, Double)], dim: Int, stepSize: Double, maxIter: Int): Unit = {
    val initWeight = new DenseVector[Double](dim)
    val sgd = StochasticGradientDescent[DenseVector[Double]](stepSize, maxIter)
    val states = sgd.iterations(Logistic.Cost(trainData), initWeight)

    val lossHistory = new ArrayBuffer[Double]()
    var weight = new DenseVector[Double](dim)
    while (states.hasNext) {
      val state = states.next()
      lossHistory += state.value

      if (!states.hasNext) {
        weight = state.x
      }
    }
    println(s"loss history: ${lossHistory.toArray.mkString(" ")}")
    println(s"weights: ${weight.toArray.mkString(" ")}")
  }

  def runPsSGD(trainData: RDD[(Vector, Double)], dim: Int, stepSize: Double, maxIter: Int): Unit = {
    val pool = PSContext.getOrCreate().createModelPool(dim, 10)
    val initWeightPS = pool.createZero().mkBreeze()
    val sgd = StochasticGradientDescent[BreezePSVector](stepSize, maxIter)
    val states = sgd.iterations(Logistic.PSCost(trainData), initWeightPS)

    val lossHistory = new ArrayBuffer[Double]()
    var weight: BreezePSVector = null
    while (states.hasNext) {
      val state = states.next()
      lossHistory += state.value

      if (!states.hasNext) {
        weight = state.x
      }
    }
    println(s"loss history: ${lossHistory.toArray.mkString(" ")}")
    println(s"weights: ${weight.toRemote.pull().mkString(" ")}")
    PSContext.getOrCreate().destroyVectorPool(pool)
  }

  def runPsAggregateSGD(
      trainData: RDD[(Vector, Double)],
      dim: Int,
      stepSize: Double,
      maxIter: Int): Unit = {

    val pool = PSContext.getOrCreate().createModelPool(dim, 10)
    val initWeightPS = pool.createZero().mkBreeze()
    val sgd = StochasticGradientDescent[BreezePSVector](stepSize, maxIter)
    val states = sgd.iterations(Logistic.PSAggregateCost(trainData), initWeightPS)

    val lossHistory = new ArrayBuffer[Double]()
    var weight: BreezePSVector = null
    while (states.hasNext) {
      val state = states.next()
      lossHistory += state.value

      if (!states.hasNext) {
        weight = state.x
      }
    }
    println(s"loss history: ${lossHistory.toArray.mkString(" ")}")
    println(s"weights: ${weight.toRemote.pull().mkString(" ")}")
    PSContext.getOrCreate().destroyVectorPool(pool)
  }

}
