import org.apache.spark.SparkContext
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity.Entropy

// Load and parse the data file
val inputData = sc.textFile("hdfs://129.207.46.225:8020/yzyan/mlclass/sample_tree_data.csv")
val mldata = inputData.map { line =>
  val parts = line.split(',').map(_.toDouble)
  LabeledPoint(parts(0), Vectors.dense(parts.tail))
}

val splits = mldata.randomSplit(Array(0.7, 0.3))
val (trainingData, parsedData) = (splits(0), splits(1))

// Run training algorithm to build the model
val maxDepth = 5
val model = DecisionTree.train(trainingData, Classification, Entropy, maxDepth)

val labelAndPreds = parsedData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}

val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / parsedData.count
println("Training Error = " + trainErr)

