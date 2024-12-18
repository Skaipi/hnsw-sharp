// <copyright file="Node.Algorithm4.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Numerics;

    internal partial class Algorithms
    {
        /// <summary>
        /// The implementation of the SELECT-NEIGHBORS-HEURISTIC(q, C, M, lc, extendCandidates, keepPrunedConnections) algorithm.
        /// Article: Section 4. Algorithm 4.
        /// </summary>
        /// <typeparam name="TItem">The typeof the items in the small world.</typeparam>
        /// <typeparam name="TDistance">The type of the distance in the small world.</typeparam>
        internal sealed class Algorithm4<TItem, TDistance> : Algorithm<TItem, TDistance> where TDistance : struct, IFloatingPoint<TDistance>
        {
            public Algorithm4(Graph<TItem, TDistance>.Core graphCore) : base(graphCore)
            {
            }

            /// <inheritdoc/>
            internal override List<int> SelectBestForConnecting(List<ValueTuple<TDistance, int>> candidatesIds, TravelingCosts<int, TDistance> travelingCosts, int layer)
            {
                /*
                 * q ← this
                 * R ← ∅    // result
                 * W ← C    // working queue for the candidates
                 * if expandCandidates  // expand candidates
                 *   for each e ∈ C
                 *     for each eadj ∈ neighbourhood(e) at layer lc
                 *       if eadj ∉ W
                 *         W ← W ⋃ eadj
                 *
                 * Wd ← ∅ // queue for the discarded candidates
                 * while │W│ gt 0 and │R│ lt M
                 *   e ← extract nearest element from W to q
                 *   if e is closer to q compared to any element from R
                 *     R ← R ⋃ e
                 *   else
                 *     Wd ← Wd ⋃ e
                 *
                 * if keepPrunedConnections // add some of the discarded connections from Wd
                 *   while │Wd│ gt 0 and │R│ lt M
                 *   R ← R ⋃ extract nearest element from Wd to q
                 *
                 * return R
                 */

                var layerM = GetM(layer);
                var resultHeap = new BinaryHeap<(TDistance, int)>(new List<ValueTuple<TDistance, int>>(layerM + 1), GraphCore.FartherIsOnTop);
                var candidatesHeap = new BinaryHeap<(TDistance, int)>(candidatesIds, GraphCore.CloserIsOnTop);

                // expand candidates option is enabled
                if (GraphCore.Parameters.ExpandBestSelection)
                {
                    var visited = new HashSet<int>(candidatesHeap.Buffer.ConvertAll(x => x.Item2));
                    var toAdd = new HashSet<ValueTuple<TDistance, int>>();
                    foreach (var candidateTuple in candidatesHeap.Buffer)
                    {
                        var candidateId = candidateTuple.Item2;
                        var candidateNeighborsIDs = GraphCore.Nodes[candidateId][layer];
                        foreach (var candidateNeighbourId in candidateNeighborsIDs)
                        {
                            if (!visited.Contains(candidateNeighbourId))
                            {
                                toAdd.Add((travelingCosts.From(candidateNeighbourId), candidateNeighbourId));
                                visited.Add(candidateNeighbourId);
                            }
                        }
                    }
                    foreach (var id in toAdd)
                    {
                        candidatesHeap.Push(id);
                    }
                }

                // main stage of moving candidates to result
                var discardedHeap = new BinaryHeap<ValueTuple<TDistance, int>>(new List<ValueTuple<TDistance, int>>(candidatesHeap.Buffer.Count), GraphCore.CloserIsOnTop);
                while (candidatesHeap.Buffer.Any() && resultHeap.Buffer.Count < layerM)
                {
                    (var candidateDist, var candidateId) = candidatesHeap.Pop();
                    (var farthestResultDist, var farthestResultId) = resultHeap.Buffer.FirstOrDefault();

                    if (!resultHeap.Buffer.Any() || DistanceUtils.LowerThan(candidateDist, farthestResultDist))
                    {
                        resultHeap.Push((candidateDist, candidateId));
                    }
                    else if (GraphCore.Parameters.KeepPrunedConnections)
                    {
                        discardedHeap.Push((candidateDist, candidateId));
                    }
                }

                // keep pruned option is enabled
                if (GraphCore.Parameters.KeepPrunedConnections)
                {
                    while (discardedHeap.Buffer.Any() && resultHeap.Count < layerM)
                    {
                        resultHeap.Push(discardedHeap.Pop());
                    }
                }

                return resultHeap.Buffer.ConvertAll(x => x.Item2);
            }
        }
    }
}
