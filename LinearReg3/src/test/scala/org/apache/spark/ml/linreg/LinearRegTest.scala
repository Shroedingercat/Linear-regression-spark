package org.apache.spark.ml.linreg

import java.nio.file.Files

import breeze.linalg.{DenseVector, InjectNumericOps}
import breeze.stats.distributions.Gaussian
import org.scalatest._
import flatspec._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql._
import org.scalatest.matchers._


class LinearRegTest extends AnyFlatSpec with should.Matchers {
  val delta = 0.0001
  lazy val data: DataFrame = LinearRegTest._data
  lazy val data_labels: DataFrame = LinearRegTest._data_labels
  "Model" should "return predictions" in {
    val model = new LinearRegressionModel(
      Vectors.dense(0.5, 0.5).toDense, 1.0
    ).setInputCol("features")
      .setOutputCol("features")
      .setRandomSeed(1)

    val vectors: Array[Vector] = model.transform(data).collect().map(_.getAs[Vector](0))

    vectors.length should be(2)
    vectors(0)(0) should be (2.0 +- delta)
    vectors(1)(0) should be (0.5*0.5 + 0.5*0.5 + 1 +- delta)
  }

  "Estimator" should "return prediction" in {
    val estimator = new LinearRegression().setInputCol("features")
      .setOutputCol("features")
      .setLabelCol("labels")
      .setMaxIter(1)
      .setRandomSeed(1)

    val model = estimator.fit(data_labels)

    val vectors: Array[Vector] = model.transform(data_labels).collect().map(_.getAs[Vector](0))

    vectors.length should be(100000)
  }

  "Estimator" should "return prediction with a smaller mse" in {
    val estimator = new LinearRegression().setInputCol("features")
      .setOutputCol("pred")
      .setLabelCol("labels")
      .setMaxIter(1)
      .setRandomSeed(1)

    val model = estimator.fit(data_labels)
    val bad_pred = model.transform(data_labels)
    val bad_mse: Double = model.evaluate(bad_pred)

    val estimator2 = new LinearRegression().setInputCol("features")
      .setOutputCol("pred")
      .setLabelCol("labels")
      .setMaxIter(1000)
      .setRandomSeed(1)
    val model2 = estimator2.fit(data_labels)
    val pred = model2.transform(data_labels)
    val mse: Double = model2.evaluate(pred)
    assert(bad_mse > mse)
  }

  // На windows не удалось решить проблему c созданием tmp folder :(
  "Estimator" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setInputCol("features")
        .setOutputCol("pred")
        .setLabelCol("labels")
        .setRandomSeed(1)
    ))

    val tmpFolder = Files.createTempDirectory("tmp").toFile
    val model = pipeline.fit(data_labels)
    val prediction: Array[Vector] = model.transform(data_labels).collect().map(_.getAs[Vector](0))
    pipeline.write.overwrite().save(tmpFolder.getAbsolutePath)

    val prediction2 = Pipeline.load(tmpFolder.getAbsolutePath).fit(data_labels)
      .transform(data_labels).collect().map(_.getAs[Vector](0))
    for (i <- 0 to (prediction.size-1)) {
      prediction(i)(0) should be (prediction2(i)(0) +- delta)
    }
  }
}


object LinearRegTest extends WithSpark {
  lazy val normal01: Gaussian = breeze.stats.distributions.Gaussian(0, 1)
  lazy val w: DenseVector[Double] = breeze.linalg.DenseVector(1.5, 0.3, -0.7)
  val b: Double = 1.0
  var xy: Array[(Vector, Vector)] = Array.empty[(Vector, Vector)]
  for (i <- 1 to 100000) {
    var rand = breeze.linalg.DenseVector.rand(3, normal01)
    var y: Double = (rand.dot(w) + b)
    xy :+= Tuple2(Vectors.fromBreeze(rand), Vectors.dense(y))
  }
  lazy val _data: DataFrame = {
    import sqlc.implicits._
    Seq(Tuple1(Vectors.dense(1, 1)),
      Tuple1(Vectors.dense(0.5, 0.5))).toDF("features")
  }
  lazy val _data_labels: DataFrame = {
    import sqlc.implicits._
    xy.toSeq.toDF("features", "labels")
  }
}








