import breeze.linalg.DenseVector
import breeze.stats.distributions.Gaussian
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.ml.linreg.LinearRegression
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions.udf
import org.apache.spark.sql.{DataFrame, SQLContext, SparkSession}

val spark: SparkSession = SparkSession.builder.appName("Simple Application")
  .master("local[4]")
  .getOrCreate()

val sqlc: SQLContext = spark.sqlContext

import sqlc.implicits._

val normal01: Gaussian = breeze.stats.distributions.Gaussian(0, 1)
val w: DenseVector[Double] = breeze.linalg.DenseVector(1.0, 2.0, 0.1)
var xy: Array[(Vector, Vector)] = Array.empty[(Vector, Vector)]
var x: Array[(Vector)] = Array.empty[(Vector)]
for (i <- 1 to 1000) {
  var rand = breeze.linalg.DenseVector.rand(3, normal01)
  var y: Double = (rand.dot(w) + 1.0)
  xy :+= Tuple2(Vectors.fromBreeze(rand), Vectors.dense(y))
  x :+= Vectors.fromBreeze(rand)
}
val data: DataFrame = x.map((x:Vector) => Tuple1(x)).toSeq.toDF("features")
val data_labels: DataFrame = xy.toSeq.toDF("features", "labels")

val estimator = new LinearRegression().setInputCol("features")
  .setOutputCol("features")
  .setLabelCol("labels")
  .setMaxIter(1000)

val model = estimator.fit(data_labels)

val vectors: Array[Double] = model.transform(data).collect().map(_.getAs[Double](0))
vectors
vectors