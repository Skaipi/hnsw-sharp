// <copyright file="SmallWorld.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Numerics;
    using System.Threading;
    using MessagePack;
    using MessagePackCompat;

    /// <summary>
    /// The Hierarchical Navigable Small World Graphs. https://arxiv.org/abs/1603.09320
    /// </summary>
    /// <typeparam name="TItem">The type of items to connect into small world.</typeparam>
    /// <typeparam name="TDistance">The type of distance between items (expect any numeric type: float, double, decimal, int, ...).</typeparam>
    public partial class SmallWorld<TItem, TDistance> where TDistance : struct, IFloatingPoint<TDistance>
    {
        private const string SERIALIZATION_HEADER = "HNSW";
        private readonly Func<TItem, TItem, TDistance> Distance;

        private Graph<TItem, TDistance> Graph;
        private IProvideRandomValues Generator;

        private ReaderWriterLockSlim _rwLock;

        /// <summary>
        /// Gets the list of items currently held by the SmallWorld graph. 
        /// The list is not protected by any locks, and should only be used when it is known the graph won't change
        /// </summary>
        public IReadOnlyDictionary<int, TItem> UnsafeItems => Graph?.GraphCore?.Items;

        /// <summary>
        /// Gets a copy of the list of items currently held by the SmallWorld graph. 
        /// This call is protected by a read-lock and is safe to be called from multiple threads.
        /// </summary>
        public IReadOnlyDictionary<int, TItem> Items
        {
            get
            {
                if (_rwLock is object)
                {
                    _rwLock.EnterReadLock();
                    try
                    {
                        return Graph.GraphCore.Items;
                    }
                    finally
                    {
                        _rwLock.ExitReadLock();
                    }
                }
                else
                {
                    return Graph?.GraphCore?.Items;
                }
            }
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="SmallWorld{TItem, TDistance}"/> class.
        /// </summary>
        /// <param name="distance">The distance function to use in the small world.</param>
        /// <param name="generator">The random number generator for building graph.</param>
        /// <param name="parameters">Parameters of the algorithm.</param>
        public SmallWorld(Func<TItem, TItem, TDistance> distance, IProvideRandomValues generator, Parameters parameters, bool threadSafe = true)
        {
            Distance = distance;
            Graph = new Graph<TItem, TDistance>(Distance, parameters);
            Generator = generator;
            _rwLock = threadSafe ? new ReaderWriterLockSlim() : null;
        }

        /// <summary>
        /// Builds hnsw graph from the items.
        /// </summary>
        /// <param name="items">The items to connect into the graph.</param>

        public IReadOnlyList<int> AddItems(IReadOnlyList<TItem> items, IProgressReporter progressReporter = null)
        {
            _rwLock?.EnterWriteLock();
            try
            {
                return Graph.AddItems(items, Generator, progressReporter);
            }
            finally
            {
                _rwLock?.ExitWriteLock();
            }
        }

        /// <summary>
        /// Removes item from the hnsw graph
        /// </summary>
        public void RemoveItem(int itemIndex)
        {
            _rwLock?.EnterWriteLock();
            try
            {
                Graph.RemoveItem(itemIndex);
            }
            finally
            {
                _rwLock?.ExitWriteLock();
            }
        }

        /// <summary>
        /// Run knn search for a given item.
        /// </summary>
        /// <param name="item">The item to search nearest neighbours.</param>
        /// <param name="k">The number of nearest neighbours.</param>
        /// <param name="filterItem">Filter results by ID that should be kept (return true to keep, false to exclude from results)</param>
        /// <param name="cancellationToken">Cancellation Token for stopping the search when filtering is active</param>
        /// <returns>The list of found nearest neighbours.</returns>
        public IList<KNNSearchResult> KNNSearch(TItem item, int k, Func<TItem, bool> filterItem = null, CancellationToken cancellationToken = default)
        {
            _rwLock?.EnterReadLock();
            try
            {
                var result = Graph.KNearest(item, Math.Max(Graph.Parameters.MinNN, k), filterItem, cancellationToken);
                if (Graph.Parameters.MinNN > k)
                {
                    return result.OrderBy(x => -x.Distance).Take(k).ToList();
                }
                return result;
            }
            finally
            {
                _rwLock?.ExitReadLock();
            }
        }

        /// <summary>
        /// Get the item with the index
        /// </summary>
        /// <param name="index">The index of the item</param>
        public TItem GetItem(int index)
        {
            _rwLock?.EnterReadLock();
            try
            {
                return Items[index];
            }
            finally
            {
                _rwLock?.ExitReadLock();
            }
        }

        /// <summary>
        /// Serializes the graph WITHOUT linked items.
        /// </summary>
        /// <returns>Bytes representing the graph.</returns>
        public void SerializeGraph(Stream stream)
        {
            if (Graph == null)
            {
                throw new InvalidOperationException("The graph does not exist");
            }
            _rwLock?.EnterReadLock();
            try
            {
                MessagePackBinary.WriteString(stream, SERIALIZATION_HEADER);
                MessagePackSerializer.Serialize(stream, Graph.Parameters);
                Graph.Serialize(stream);
            }
            finally
            {
                _rwLock?.ExitReadLock();
            }
        }

        /// <summary>
        /// Deserializes the graph from byte array.
        /// </summary>
        /// <param name="items">The items to assign to the graph's verticies.</param>
        /// <param name="bytes">The serialized parameters and edges.</param>
        public static SmallWorld<TItem, TDistance> DeserializeGraph(IReadOnlyList<TItem> items, Func<TItem, TItem, TDistance> distance, IProvideRandomValues generator, Stream stream, bool threadSafe = true)
        {
            var p0 = stream.Position;
            string hnswHeader;
            try
            {
                hnswHeader = MessagePackBinary.ReadString(stream);
            }
            catch (Exception E)
            {
                if (stream.CanSeek) { stream.Position = p0; } //Resets the stream to original position
                throw new InvalidDataException($"Invalid header found in stream, data is corrupted or invalid", E);
            }

            if (hnswHeader != SERIALIZATION_HEADER)
            {
                if (stream.CanSeek) { stream.Position = p0; } //Resets the stream to original position
                throw new InvalidDataException($"Invalid header found in stream, data is corrupted or invalid");
            }

            // readStrict: true -> removed, as not available anymore on MessagePack 2.0 - also probably not necessary anymore
            //                     see https://github.com/neuecc/MessagePack-CSharp/pull/663

            var parameters = MessagePackSerializer.Deserialize<Parameters>(stream);

            var world = new SmallWorld<TItem, TDistance>(distance, generator, parameters, threadSafe: threadSafe);
            world.Graph.Deserialize(items, stream);
            return world;
        }

        /// <summary>
        /// Prints edges of the graph. Mostly for debug and test purposes.
        /// </summary>
        /// <returns>String representation of the graph's edges.</returns>
        public string Print()
        {
            return Graph.Print();
        }

        public void PrintStats()
        {
            var maxLayer = Graph.MaxLayer;
            Console.WriteLine($"Graph max layer: {maxLayer}\n");

            for (int layer = maxLayer - 1; layer >= 0; layer--)
            {
                var M = Graph.GraphCore.Algorithm.GetM(layer);
                var nodesOnLayer = Graph.GraphCore.Nodes.Where(x => x.MaxLayer >= layer && !Graph.GraphCore.RemovedIndexes.Contains(x.Id)).ToList();
                Console.WriteLine($"Nodes on layer {layer}: {nodesOnLayer.Count}");

                var minOutConn = nodesOnLayer.Min(x => x.Connections[layer].Count);
                var maxOutConn = nodesOnLayer.Max(x => x.Connections[layer].Count);
                var avgOutConn = nodesOnLayer.Average(x => x.Connections[layer].Count);
                var outConnAboveM = nodesOnLayer.Where(x => x.Connections[layer].Count == M).ToList().Count;

                Console.WriteLine($"  Minimal outgoing connections: {minOutConn}");
                Console.WriteLine($"  Maximal outgoing connections: {maxOutConn}");
                Console.WriteLine($"  Average outgoing connections: {avgOutConn}");
                Console.WriteLine($"  Elements with numeber of outgoing equal M: {outConnAboveM}");
                Console.WriteLine();

                var minInConn = nodesOnLayer.Min(x => x.InConnections[layer].Count);
                var maxInConn = nodesOnLayer.Max(x => x.InConnections[layer].Count);
                var avgInConn = nodesOnLayer.Average(x => x.InConnections[layer].Count);
                var inConnAboveM = nodesOnLayer.Where(x => x.InConnections[layer].Count > M).ToList().Count;

                Console.WriteLine($"  Minimal incoming connections: {minInConn}");
                Console.WriteLine($"  Maximal incoming connections: {maxInConn}");
                Console.WriteLine($"  Average incoming connections: {avgInConn}");
                Console.WriteLine($"  Elements with numeber of incoming connections above M: {inConnAboveM}");
                Console.WriteLine();
            }
        }

        [MessagePackObject(keyAsPropertyName: true)]
        public class Parameters
        {
            public Parameters()
            {
                M = 10;
                LevelLambda = 1 / Math.Log(M);
                NeighbourHeuristic = NeighbourSelectionHeuristic.SelectSimple;
                ConstructionPruning = 200;
                ExpandBestSelection = false;
                KeepPrunedConnections = false;
                InitialItemsSize = 1024;
            }

            /// <summary>
            /// Gets or sets the parameter which defines the maximum number of neighbors in the zero and above-zero layers.
            /// The maximum number of neighbors for the zero layer is 2 * M.
            /// The maximum number of neighbors for higher layers is M.
            /// </summary>
            public int M { get; set; }

            /// <summary>
            /// Gets or sets the max level decay parameter. https://en.wikipedia.org/wiki/Exponential_distribution See 'mL' parameter in the HNSW article.
            /// </summary>
            public double LevelLambda { get; set; }

            /// <summary>
            /// Gets or sets parameter which specifies the type of heuristic to use for best neighbours selection.
            /// </summary>
            public NeighbourSelectionHeuristic NeighbourHeuristic { get; set; }

            /// <summary>
            /// Gets or sets the number of candidates to consider as neighbours for a given node at the graph construction phase. See 'efConstruction' parameter in the article.
            /// </summary>
            public int ConstructionPruning { get; set; }

            /// <summary>
            /// Gets or sets the minimal number of nodes obtained by knn search. If provided k exceeds this value, the search result will be trimmed to k. Improves recall for small k.
            /// </summary>
            public int MinNN { get; set; }

            /// <summary>
            /// Gets or sets a value indicating whether to expand candidates if <see cref="NeighbourSelectionHeuristic.SelectHeuristic"/> is used. See 'extendCandidates' parameter in the article.
            /// </summary>
            public bool ExpandBestSelection { get; set; }

            /// <summary>
            /// Gets or sets a value indicating whether to keep pruned candidates if <see cref="NeighbourSelectionHeuristic.SelectHeuristic"/> is used. See 'keepPrunedConnections' parameter in the article.
            /// </summary>
            public bool KeepPrunedConnections { get; set; }

            /// <summary>
            /// Gets or sets a the initial size of the Items list
            /// </summary>
            public int InitialItemsSize { get; set; }
        }

        public class KNNSearchResult
        {
            internal KNNSearchResult(int id, TItem item, TDistance distance)
            {
                Id = id;
                Item = item;
                Distance = distance;
            }

            public int Id { get; }

            public TItem Item { get; }

            public TDistance Distance { get; }

            public override string ToString()
            {
                return $"I:{Id} Dist:{Distance:n2} [{Item}]";
            }
        }
    }
}
