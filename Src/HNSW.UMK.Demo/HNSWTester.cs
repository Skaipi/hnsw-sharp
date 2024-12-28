using HNSW.Net;
using System.Linq;
using System.Xml.Serialization;

namespace HNSW.UMK.Demo;

public class HNSWTester
{
    public List<float[]> Vectors { get; private set; }
    public List<bool> RemovedVectors { get; private set; }
    private const int defaultVectorsAmount = 20_000;
    private const int efConstruction = 100;
    private const int defaultM = 16;
    private float TotalSelfRecallError = 0;
    private int _threads;
    public SmallWorld<float[], float>? graph;
    private SmallWorld<float[], float>.Parameters parameters;

    public HNSWTester(List<float[]> vectors, int itemsAmount = defaultVectorsAmount, int ef_construction = efConstruction, int M = defaultM, int threads = 1)
    {
        Vectors = vectors;
        RemovedVectors = new List<bool>(vectors.Count);
        for (int i = 0; i < vectors.Count; i++)
        {
            RemovedVectors.Add(false);
        }
        parameters = new SmallWorld<float[], float>.Parameters()
        {
            M = M,
            LevelLambda = 1 / Math.Log(M),
            NeighbourHeuristic = NeighbourSelectionHeuristic.CustomHeuristic,
            ConstructionPruning = ef_construction,
            ExpandBestSelection = false,
            KeepPrunedConnections = false,
            InitialItemsSize = itemsAmount,
        };

        _threads = threads;
    }

    public IReadOnlyList<int> BuildGraph()
    {
        var threadSafe = _threads > 1;
        graph = new SmallWorld<float[], float>(CosineDistance.SIMD, DefaultRandomGenerator.Instance, parameters, threadSafe);
        var chunks = SplitList(Vectors, _threads);

        if (_threads > 1)
        {
            List<Task> tasks = new List<Task>();
            for (int i = 0; i < _threads; i++)
            {
                tasks.Add(Task.Run(() => graph.AddItems(chunks[i])));
            }
            Task.WaitAll(tasks.ToArray());
            return new List<int>();
        }
        else
        {
            return graph.AddItems(Vectors);
        }
    }

    public void ConnectionStats()
    {
        if (graph is null)
            return;

        Console.WriteLine();
        graph.PrintStats();
    }

    public void RemoveVector(int vecId)
    {
        if (graph is null)
        {
            throw new Exception("Remove on ininitialized graph");
        }
        graph.RemoveItem(vecId);
        RemovedVectors[vecId] = true;
    }

    public void SelfRecall()
    {
        if (graph is null)
        {
            throw new Exception("Search operation on uninitialized graph");
        }

        int bestWrong = 0;
        if (_threads > 1)
        {

        }
        else
        {
            for (int i = 0; i < Vectors.Count; ++i)
            {
                if (RemovedVectors[i]) continue;
                var result = graph.KNNSearch(Vectors[i], 1);
                var best = result.OrderBy(r => r.Distance).First();
                if (best.Id != i)
                {
                    bestWrong++;
                }
                TotalSelfRecallError += best.Distance;
            }
        }
        Console.WriteLine($"Wrongly found nearest neighbours: {bestWrong}");
        Console.WriteLine($"Average distance from nearest neighbour: {TotalSelfRecallError / Vectors.Count}");
    }

    private List<List<T>> SplitList<T>(List<T> list, int k)
    {
        int n = list.Count;
        var chunks = new List<List<T>>();
        int minSize = n / k;
        int remainder = n % k;

        int start = 0;
        for (int i = 0; i < k; i++)
        {
            int chunkSize = minSize + (i < remainder ? 1 : 0);
            chunks.Add(list.GetRange(start, chunkSize));
            start += chunkSize;
        }
        return chunks;
    }
}
