// See https://aka.ms/new-console-template for more information

using HNSW.Net;
using HNSW.UMK.Demo;

static List<float[]> RandomVectors(int vectorSize, int vectorsCount)
{
  var vectors = new List<float[]>();

  for (int i = 0; i < vectorsCount; i++)
  {
    var vector = new float[vectorSize];
    DefaultRandomGenerator.Instance.NextFloats(vector);
    VectorUtils.NormalizeSIMD(vector);
    vectors.Add(vector);
  }

  return vectors;
}

int num_vectors = 10_000;

var vectors = FVECReader.ReadFVEC("C:/Users/skaii/PhD/knn/hnswlib-python/sift/sift_base.fvecs", num_vectors);
for (int i = 0; i < vectors.Count; i++)
{
  VectorUtils.NormalizeSIMD(vectors[i]);
}

// var vectors = RandomVectors(128, num_vectors);
Console.WriteLine($"{vectors.Count} Vectors loaded from the disc");

HNSWTester tester = new HNSWTester(vectors, num_vectors);
var ids = tester.BuildGraph();
Console.WriteLine("Pre-removal search:");
tester.SelfRecall();
tester.ConnectionStats();

for (int i = 0; i < num_vectors; i += 2)
{
  Console.Write($"{i}/{num_vectors}\r");
  tester.RemoveVector(i);
}

Console.WriteLine($"Post-removal search:");
tester.SelfRecall();
tester.ConnectionStats();