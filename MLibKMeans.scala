import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.mllib.linalg.Vectors

val inputData = sc.textFile("hdfs://129.207.46.225:8020/mlclass/kmeans_long.csv")


val parsedData = inputData.map(s => Vectors.dense(s.split(',').map(_.toDouble)))

val numClusters = 2
val numIterations = 20
val clusters = KMeans.train(parsedData, numClusters, numIterations)


val WSSSE = clusters.computeCost(parsedData)
println("WSSSE = " + WSSSE)


// clusters.k
// clusters.clusterCenters.zipWithIndex

val results = parsedData.map(point=> {val prediction=clusters.predict(point);(point.toString,prediction)})
print(results.collect().toList);

clusters.predict(Vectors.dense(Array(60.0,104.0)))

clusters.predict(Vectors.dense(Array(40.0,54.0)))


//val inputData = sc.textFile("hdfs://129.207.46.225:8020/mlclass/kmeans_short.csv")
//val inputData = sc.textFile("hdfs://129.207.46.225:8020/mlclass/kmeans_iris.csv")
