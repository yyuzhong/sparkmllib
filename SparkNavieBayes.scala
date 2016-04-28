import org.apache.spark.SparkContext
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity.Entropy
import org.apache.spark.mllib.classification.NaiveBayes

// Load and parse the data file
val tData = sc.textFile("hdfs://129.207.46.225:8020/mlclass/v100k.csv")
//val tData = sc.textFile("hdfs://129.207.46.225:8020/mlclass/v10m.csv")
//val tData = sc.textFile("hdfs://129.207.46.225:8020/mlclass/v100m.csv")

val trainningData = tData.filter(line => !line.contains("%")).map { line =>
  val parts = line.split(',').map(_.toDouble)
  LabeledPoint(parts(3), Vectors.dense(parts.take(3)))
}

val pData = sc.textFile("hdfs://129.207.46.225:8020/mlclass/vehicledata.csv")
val predictData = pData.filter(line => !line.contains("%")).map { line =>
  val parts = line.split(',').map(_.toDouble)
  LabeledPoint(parts(3), Vectors.dense(parts.take(3)))
}

//val splits = mldata.randomSplit(Array(0.7, 0.3))
//val (trainingData, parsedData) = (splits(0), splits(1))

// Run training algorithm to build the model
val maxDepth = 5
//val model = DecisionTree.train(trainningData, Classification, Entropy, maxDepth)
val model = NaiveBayes.train(trainningData, lambda = 1.0)

val labelAndPreds = predictData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}

val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / predictData.count
println("Training Error = " + trainErr)

