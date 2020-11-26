package org.apache.spark.ml.linreg

import breeze.stats.distributions.{RandBasis, ThreadLocalRandomGenerator}
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{Param, ParamMap, Params}
import org.apache.spark.ml.param.shared.{HasInputCol, HasLabelCol, HasMaxIter, HasOutputCol}
import org.apache.spark.ml.util.{DatasetUtils, DefaultParamsReadable, DefaultParamsWritable, Identifiable, SchemaUtils}
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.{DataFrame, Dataset, Encoder, Row}
import org.apache.spark.sql.types.StructType
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer

trait LinearRegressionParams extends HasInputCol with HasOutputCol with HasLabelCol with Params with HasMaxIter {
  val metric: Param[String] = new Param[String](this, "metric", "metric for evaluation")
  val seed: Param[Int] = new Param[Int](this, "seed", "random seed")
  val learningRate: Param[Double] = new Param[Double](this, "learningRate", "learningRate")
  def setMetric(str: String = "MSE"):this.type = set(metric, str)
  def setLearningRate(double: Double = 0.001):this.type = set(learningRate, double)
  def setInputCol(str: String):this.type = set(inputCol, str)
  def setOutputCol(str: String):this.type = set(outputCol, str)
  def setLabelCol(str: String):this.type = set(labelCol, str)
  def setMaxIter(int: Int):this.type = set(maxIter, int)
  def setRandomSeed(int: Int):this.type = set(seed, int)

  setDefault(maxIter -> 100, learningRate -> 1E-6)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())

    if (schema.fieldNames.contains($(outputCol))) {
      SchemaUtils.checkColumnType(schema, getOutputCol, new VectorUDT())
      schema
    } else {
      SchemaUtils.appendColumn(schema, schema(getInputCol).copy(name = getOutputCol))
    }
  }
}

/**
 * TO DO:
 * @param uid
 */
class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel]
  with LinearRegressionParams with DefaultParamsWritable{
  def this() = this(Identifiable.randomUID("LinearRegression_"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {
    implicit val encoder: Encoder[Vector] = ExpressionEncoder()
    val dim = dataset.select($(inputCol)).rdd.first().toString().split(",").size

    implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister($(seed))))
    val normal01 = breeze.stats.distributions.Gaussian(0, 1)
    var weights = breeze.linalg.DenseVector.rand(dim, normal01)
    val r = scala.util.Random
    var bayes = r.nextDouble()

    val columnNames = Seq[String]($(inputCol), $(labelCol))
    val colNames = columnNames.map(name => dataset(name))

    val vectors = dataset.select(colNames:_*)
      .withColumn($(inputCol), dataset($(inputCol)).as[Vector])
      .rdd.map{x:Row => Array(x.getAs[Vector](0), x.getAs[Vector](1))}

    val size: Double = vectors.count().toDouble
    val getGradW = (x: Array[Vector]) => x(0).asBreeze *:* (2.0 / size) * $(learningRate) * (x(0).dot(Vectors.fromBreeze(weights)) + bayes - x(1)(0))
    val getGradb = (x: Array[Vector]) => ((2.0/size) *$(learningRate)*(x(0).dot(Vectors.fromBreeze(weights)) + bayes - x(1)(0)))
      for (i <- 1 to $(maxIter)) {
        var grad: breeze.linalg.Vector[Double] = vectors
          .mapPartitions{data:Iterator[Array[Vector]]=> for (x <- data) yield getGradW(x)
          }.reduce(_ + _)

        var grad_b: Double = vectors
          .mapPartitions{data:Iterator[Array[Vector]]=> for (x <- data) yield getGradb(x)
          }.reduce(_ + _)

        weights = weights - grad
        bayes -= grad_b
    }

    copyValues(new LinearRegressionModel(Vectors.fromBreeze(weights).toDense, bayes).setParent(this))
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object StandardScaler extends DefaultParamsReadable[LinearRegression]

/**
 * TO DO:
 * @param uid
 * @param weights weights
 */
class LinearRegressionModel private[linreg] (override val uid: String, val weights: DenseVector, val bayes: Double)
  extends Model[LinearRegressionModel] with LinearRegressionParams {

  private[linreg] def this(weights: DenseVector, bayes: Double) =
    this(Identifiable.randomUID("LinearRegression_"), weights, bayes)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(new LinearRegressionModel(weights, bayes))

  def predict(X: DenseVector):Vector = {
    Vectors.dense(X.dot(weights) + bayes)
  }

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf = dataset.sqlContext.udf.register(uid + "_transform",
      (x : DenseVector) => {predict(x)})
    dataset.withColumn($(outputCol), transformUdf(dataset($(inputCol))))

  }

  def evaluate(dataset: Dataset[_]): Double ={
    implicit val encoder: Encoder[Vector] = ExpressionEncoder()
    val columnNames = Seq[String]($(outputCol), $(labelCol))
    val colNames = columnNames.map(name => dataset(name))

    val vectors = dataset.select(colNames:_*)
      .withColumn($(outputCol), dataset($(outputCol)).as[Vector])
      .withColumn($(labelCol), dataset($(labelCol)).as[Vector])
      .rdd.map{x:Row => Array(x.getAs[Vector](0), x.getAs[Vector](1))}
    val size: Double = vectors.count().toDouble
    vectors.map{x:Array[Vector] => (x(0)(0) - x(1)(0))*(x(0)(0) - x(1)(0)/size)}
      .reduce(_ + _)
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}