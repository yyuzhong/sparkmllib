import org.apache.spark.SparkContext
import org.apache.spark.mllib.tree.DecisionTree
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.tree.configuration.Algo._
import org.apache.spark.mllib.tree.impurity.Entropy

// Load and parse the data file
val tData = sc.textFile("hdfs://129.207.46.225:8020/mlclass/mushroom.training.csv")
val trainningData = tData.filter(line => !line.contains("%")).map { line =>
  var parts = line.split(',').map(str=>{ val cs = str.toCharArray; val c = cs(0); (c-'a').toDouble})
  if(parts(0)<10) parts(0) = 1.0;
  else parts(0) = 0.0;
  LabeledPoint(parts(0), Vectors.dense(parts.tail))
}

val pData = sc.textFile("hdfs://129.207.46.225:8020/mlclass/mushroom.test.csv")
val predictData = pData.filter(line => !line.contains("%")).map { line =>
  var parts = line.split(',').map(str=>{ val cs = str.toCharArray; val c = cs(0); (c-'a').toDouble})
  if(parts(0)<10) parts(0) = 1.0;
  else parts(0) = 0.0;
  LabeledPoint(parts(0), Vectors.dense(parts.tail))
}

// Run training algorithm to build the model
val maxDepth = 5
val model = DecisionTree.train(trainningData, Classification, Entropy, maxDepth)

val labelAndPreds = predictData.map { point =>
  val prediction = model.predict(point.features)
  (point.label, prediction)
}

val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / predictData.count
println("Training Error = " + trainErr)

