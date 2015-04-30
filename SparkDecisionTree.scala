import org.apache.spark.SparkContext
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity.Entropy
import java.util.{Calendar, Date}

val before = Calendar.getInstance().getTime();

// Load and parse the data file
val inputData = sc.textFile("hdfs://129.207.46.225:8020/mlclass/ladygaga.csv")
val mldata = inputData.filter(line => !line.contains("%")).map { line =>
  val parts = line.split(',').map(_.toDouble)
  LabeledPoint(parts(4), Vectors.dense(parts.take(4)))
}

mldata.collect().toList

//val splits = mldata.randomSplit(Array(0.7, 0.3))
//val (trainingData, parsedData) = (splits(0), splits(1))

// Run training algorithm to build the model
val maxDepth = 5
val model = DecisionTree.train(mldata, Classification, Entropy, maxDepth)

val labelAndPreds = mldata.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}

val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / mldata.count
println("Training Error = " + trainErr)

val after = Calendar.getInstance().getTime();
