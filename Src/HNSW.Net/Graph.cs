// <copyright file="Graph.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using MessagePack;
    using System.Text;

    using static HNSW.Net.EventSources;
    using System.Threading;
    using System.Numerics;

    /// <summary>
    /// The implementation of a hierarchical small world graph.
    /// </summary>
    /// <typeparam name="TItem">The type of items to connect into small world.</typeparam>
    /// <typeparam name="TDistance">The type of distance between items (expects any numeric type: float, double, decimal, int, ...).</typeparam>
    internal partial class Graph<TItem, TDistance> where TDistance : struct, IFloatingPoint<TDistance>
    {
        private readonly Func<TItem, TItem, TDistance> Distance;

        internal Core GraphCore;

        private Node? EntryPoint;

        internal int MaxLayer => EntryPoint?.MaxLayer ?? 0;

        private long _version;

        /// <summary>
        /// Initializes a new instance of the <see cref="Graph{TItem, TDistance}"/> class.
        /// </summary>
        /// <param name="distance">The distance function.</param>
        /// <param name="parameters">The parameters of the world.</param>
        internal Graph(Func<TItem, TItem, TDistance> distance, SmallWorld<TItem, TDistance>.Parameters parameters)
        {
            Distance = distance;
            Parameters = parameters;
        }

        internal SmallWorld<TItem, TDistance>.Parameters Parameters { get; }

        /// <summary>
        /// Creates graph from the given items.
        /// Contains implementation of INSERT(hnsw, q, M, Mmax, efConstruction, mL) algorithm.
        /// Article: Section 4. Algorithm 1.
        /// </summary>
        /// <param name="items">The items to insert.</param>
        /// <param name="generator">The random number generator to distribute nodes across layers.</param>
        /// <param name="progressReporter">Interface to report progress </param>
        internal IReadOnlyList<int> AddItems(IReadOnlyList<TItem> items, IProvideRandomValues generator, IProgressReporter progressReporter)
        {
            if (items is null || !items.Any()) { return Array.Empty<int>(); }

            GraphCore = GraphCore ?? new Core(Distance, Parameters);

            int startIndex = GraphCore.Items.Count;

            var newIDs = GraphCore.AddItems(items, generator);

            var entryPoint = EntryPoint ?? GraphCore.Nodes[0];

            var searcher = new Searcher(GraphCore);
            Func<int, int, TDistance> nodeDistance = GraphCore.GetDistance;

            for (int nodeId = startIndex; nodeId < GraphCore.Nodes.Count; ++nodeId)
            {
                var versionNow = Interlocked.Increment(ref _version);

                using (new ScopeLatencyTracker(GraphBuildEventSource.Instance?.GraphInsertNodeLatencyReporter))
                {
                    /*
                     * W ← ∅ // list for the currently found nearest elements
                     * ep ← get enter point for hnsw
                     * L ← level of ep // top layer for hnsw
                     * l ← ⌊-ln(unif(0..1))∙mL⌋ // new element’s level
                     * for lc ← L … l+1
                     *   W ← SEARCH-LAYER(q, ep, ef=1, lc)
                     *   ep ← get the nearest element from W to q
                     * for lc ← min(L, l) … 0
                     *   W ← SEARCH-LAYER(q, ep, efConstruction, lc)
                     *   neighbors ← SELECT-NEIGHBORS(q, W, M, lc) // alg. 3 or alg. 4
                     *     for each e ∈ neighbors // shrink connections if needed
                     *       eConn ← neighbourhood(e) at layer lc
                     *       if │eConn│ > Mmax // shrink connections of e if lc = 0 then Mmax = Mmax0
                     *         eNewConn ← SELECT-NEIGHBORS(e, eConn, Mmax, lc) // alg. 3 or alg. 4
                     *         set neighbourhood(e) at layer lc to eNewConn
                     *   ep ← W
                     * if l > L
                     *   set enter point for hnsw to q
                     */

                    // zoom in and find the best peer on the same level as newNode
                    var currentNode = GraphCore.Nodes[nodeId];
                    var currentNodeTravelingCosts = new TravelingCosts<int, TDistance>(nodeDistance, nodeId);
                    var bestPeer = FindEntryPoint(currentNode.MaxLayer, currentNodeTravelingCosts);

                    // connecting new node to the small world
                    for (int layer = Math.Min(currentNode.MaxLayer, entryPoint.MaxLayer); layer >= 0; --layer)
                    {
                        var topCandidates = searcher.RunKnnAtLayer(bestPeer.Id, currentNodeTravelingCosts, layer, Parameters.ConstructionPruning, ref _version, versionNow, _ => true);
                        var bestNeighboursIds = GraphCore.Algorithm.SelectBestForConnecting(topCandidates, currentNodeTravelingCosts, layer);
                        bestPeer = GraphCore.Nodes[bestNeighboursIds[0]]; // Best peer is at top of the heap

                        for (int i = 0; i < bestNeighboursIds.Count; ++i)
                        {
                            int newNeighbourId = bestNeighboursIds[i];

                            versionNow = Interlocked.Increment(ref _version);
                            GraphCore.Algorithm.Connect(currentNode, GraphCore.Nodes[newNeighbourId], layer);

                            versionNow = Interlocked.Increment(ref _version);
                            GraphCore.Algorithm.Connect(GraphCore.Nodes[newNeighbourId], currentNode, layer);
                        }
                    }

                    // zoom out to the highest level
                    if (currentNode.MaxLayer > entryPoint.MaxLayer)
                    {
                        entryPoint = currentNode;
                        EntryPoint = entryPoint;
                    }

                    // report distance cache hit rate
                    GraphBuildEventSource.Instance?.CoreGetDistanceCacheHitRateReporter?.Invoke(GraphCore.DistanceCacheHitRate);
                }
                progressReporter?.Progress(nodeId - startIndex, GraphCore.Nodes.Count - startIndex);
            }

            return newIDs;
        }

        internal void RecomputeItemAtLayer(int nodeId, int layer)
        {
            var searcher = new Searcher(GraphCore);
            Func<int, int, TDistance> nodeDistance = GraphCore.GetDistance;
            var bestPeer = EntryPoint ?? GraphCore.Nodes[0];

            var currentNode = GraphCore.Nodes[nodeId];
            var currentNodeTravelingCosts = new TravelingCosts<int, TDistance>(nodeDistance, nodeId);

            bestPeer = FindEntryPoint(layer, currentNodeTravelingCosts);

            var versionNow = Interlocked.Increment(ref _version);
            var topCandidates = searcher.RunKnnAtLayer(bestPeer.Id, currentNodeTravelingCosts, layer, Parameters.ConstructionPruning, ref _version, versionNow, _ => true);
            var bestNeighboursIds = GraphCore.Algorithm.SelectBestForConnecting(topCandidates, currentNodeTravelingCosts, layer);

            for (int i = 0; i < bestNeighboursIds.Count; ++i)
            {
                int newNeighbourId = bestNeighboursIds[i];
                versionNow = Interlocked.Increment(ref _version);
                GraphCore.Algorithm.Connect(currentNode, GraphCore.Nodes[newNeighbourId], layer);

                versionNow = Interlocked.Increment(ref _version);
                GraphCore.Algorithm.Connect(GraphCore.Nodes[newNeighbourId], currentNode, layer);
            }
        }

        /// <summary>
        /// Removes items from the graph.
        /// </summary>
        /// <returns></returns>
        internal int RemoveItem(int itemIndex)
        {
            if (GraphCore is null) { throw new Exception("Graph Core set to null reference"); }

            var node = GraphCore.Nodes[itemIndex];
            if (EntryPoint?.Id == itemIndex)
            {
                for (int layer = node.MaxLayer; layer >= 0; layer--)
                {
                    if (node[layer].Count > 0)
                    {
                        var neighbourId = node[layer][0];
                        EntryPoint = GraphCore.Nodes[neighbourId];
                    }
                }
            }

            for (int layer_id = 0; layer_id < node.InConnections.Count; layer_id++)
            {
                for (int j = 0; j < node.InConnections[layer_id].Count; j++)
                {
                    var neighbourId = node.InConnections[layer_id][j];
                    var neighbourNode = GraphCore.Nodes[neighbourId];
                    GraphCore.Algorithm.Disconnect(neighbourNode, node, layer_id);
                    if (neighbourNode.Connections[layer_id].Count < GraphCore.Algorithm.GetM(layer_id) / 2)
                    {
                        RecomputeItemAtLayer(neighbourId, layer_id);
                    }
                }
            }
            // No need to remove item from Nodes as it lost all connections
            GraphCore.RemovedIndexes.Add(itemIndex);

            return itemIndex;
        }

        /// <summary>
        /// Get k nearest items for a given one.
        /// Contains implementation of K-NN-SEARCH(hnsw, q, K, ef) algorithm.
        /// Article: Section 4. Algorithm 5.
        /// </summary>
        /// <param name="destination">The given node to get the nearest neighbourhood for.</param>
        /// <param name="k">The size of the neighbourhood.</param>
        /// <param name="filterItem">Filter results by ID that should be kept (return true to keep, false to exclude from results)</param>
        /// <param name="cancellationToken">Cancellation Token for stopping the search when filtering is active</param>
        /// <returns>The list of the nearest neighbours.</returns>
        internal IList<SmallWorld<TItem, TDistance>.KNNSearchResult> KNearest(TItem destination, int k, Func<TItem, bool> filterItem = null, CancellationToken cancellationToken = default)
        {
            if (EntryPoint is null) return null;

            Func<int, bool> keepResultInner = _ => true;

            if (filterItem is object)
            {
                var keepResults = new Dictionary<int, bool>();
                keepResultInner = (id) =>
                {
                    if (keepResults.TryGetValue(id, out var v)) return v;
                    v = filterItem(GraphCore.Items[id]);
                    keepResults[id] = v;
                    return v;
                };
            }

            int retries = 1_024;

            // TODO: hack we know that destination id is -1.
            TDistance RuntimeDistance(int x, int y)
            {
                int nodeId = x >= 0 ? x : y;
                return Distance(destination, GraphCore.Items[nodeId]);
            }

            while (true)
            {
                var versionNow = Interlocked.Read(ref _version);

                try
                {
                    using (new ScopeLatencyTracker(GraphSearchEventSource.Instance?.GraphKNearestLatencyReporter))
                    {
                        var searcher = new Searcher(GraphCore);
                        var destinationTravelingCosts = new TravelingCosts<int, TDistance>(RuntimeDistance, -1);
                        var bestPeer = FindEntryPoint(0, destinationTravelingCosts);
                        int visitedNodesCount = 0;

                        var topCandidates = searcher.RunKnnAtLayer(bestPeer.Id, destinationTravelingCosts, 0, k, ref _version, versionNow, keepResultInner, cancellationToken);

                        GraphSearchEventSource.Instance?.GraphKNearestVisitedNodesReporter?.Invoke(visitedNodesCount);

                        return topCandidates.ConvertAll(c => new SmallWorld<TItem, TDistance>.KNNSearchResult(c.Item2, GraphCore.Items[c.Item2], c.Item1)).ToList();
                    }
                }
                catch (GraphChangedException)
                {
                    if (retries > 0)
                    {
                        retries--;
                        continue;
                    }
                    throw;
                }
                catch (Exception)
                {
                    if (versionNow != Interlocked.Read(ref _version))
                    {
                        if (retries > 0)
                        {
                            retries--;
                            continue;
                        }
                    }
                    throw;
                }
            }
        }

        private Node FindEntryPoint(int dstLayer, TravelingCosts<int, TDistance> dstTravelingCost)
        {
            var bestPeer = EntryPoint ?? GraphCore.Nodes[0];
            var currDist = dstTravelingCost.From(bestPeer.Id);

            for (int level = bestPeer.MaxLayer; level > dstLayer; level--)
            {
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    List<int> connections = bestPeer.Connections[level];
                    int size = connections.Count;

                    for (int i = 0; i < size; i++)
                    {
                        int cand = connections[i];
                        var d = dstTravelingCost.From(cand);
                        if (DistanceUtils.LowerThan(d, currDist))
                        {
                            currDist = d;
                            bestPeer = GraphCore.Nodes[cand];
                            changed = true;
                        }
                    }
                }
            }

            return bestPeer;
        }

        /// <summary>
        /// Serializes core of the graph.
        /// </summary>
        /// <returns>Bytes representing edges.</returns>
        internal void Serialize(Stream stream)
        {
            GraphCore.Serialize(stream);
            MessagePackSerializer.Serialize(stream, EntryPoint);
        }

        /// <summary>
        /// Deserilaizes graph edges and assigns nodes to the items.
        /// </summary>
        /// <param name="items">The underlying items.</param>
        /// <param name="bytes">The serialized edges.</param>
        internal void Deserialize(IReadOnlyList<TItem> items, Stream stream)
        {
            // readStrict: true -> removed, as not available anymore on MessagePack 2.0 - also probably not necessary anymore
            //                     see https://github.com/neuecc/MessagePack-CSharp/pull/663

            var core = new Core(Distance, Parameters);
            core.Deserialize(items, stream);
            EntryPoint = MessagePackSerializer.Deserialize<Node>(stream);
            GraphCore = core;
        }

        /// <summary>
        /// Prints edges of the graph.
        /// </summary>
        /// <returns>String representation of the graph's edges.</returns>
        internal string Print()
        {
            var buffer = new StringBuilder();
            for (int layer = EntryPoint.Value.MaxLayer; layer >= 0; --layer)
            {
                buffer.AppendLine($"[LEVEL {layer}]");
                BFS(GraphCore, EntryPoint.Value, layer, (node) =>
                {
                    var neighbours = string.Join(", ", node[layer]);
                    buffer.AppendLine($"({node.Id}) -> {{{neighbours}}}");
                });

                buffer.AppendLine();
            }

            return buffer.ToString();
        }
    }
}
