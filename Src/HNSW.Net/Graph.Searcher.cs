// <copyright file="Graph.Searcher.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;
    using System.Threading;

    /// <content>
    /// The implementation of knn search.
    /// </content>
    internal partial class Graph<TItem, TDistance>
    {
        /// <summary>
        /// The graph searcher.
        /// </summary>
        internal struct Searcher
        {
            private readonly Core Core;
            private readonly List<ValueTuple<TDistance, int>> ExpansionBuffer;
            private readonly VisitedBitSet VisitedSet;



            /// <summary>
            /// Initializes a new instance of the <see cref="Searcher"/> struct.
            /// </summary>
            /// <param name="core">The core of the graph.</param>
            internal Searcher(Core core)
            {
                Core = core;
                ExpansionBuffer = new List<ValueTuple<TDistance, int>>();
                VisitedSet = new VisitedBitSet(core.Nodes.Count);
            }

            /// <summary>
            /// The implementaiton of SEARCH-LAYER(q, ep, ef, lc) algorithm.
            /// Article: Section 4. Algorithm 2.
            /// </summary>
            /// <param name="entryPointId">The identifier of the entry point for the search.</param>
            /// <param name="targetCosts">The traveling costs for the search target.</param>
            /// <param name="resultList">The list of identifiers of the nearest neighbours at the level.</param>
            /// <param name="layer">The layer to perform search at.</param>
            /// <param name="k">The number of the nearest neighbours to get from the layer.</param>
            /// <param name="version">The version of the graph, will retry the search if the version changed</param>
            /// <returns>The number of expanded nodes during the run.</returns>
            internal List<ValueTuple<TDistance, int>> RunKnnAtLayer(int entryPointId, TravelingCosts<int, TDistance> travelingCosts, int layer, int k, ref long version, long versionAtStart, Func<int, bool> keepResult, CancellationToken cancellationToken = default)
            {
                /*
                 * v ← ep // set of visited elements
                 * C ← ep // set of candidates
                 * W ← ep // dynamic list of found nearest neighbors
                 * while │C│ > 0
                 *   c ← extract nearest element from C to q
                 *   f ← get furthest element from W to q
                 *   if distance(c, q) > distance(f, q)
                 *     break // all elements in W are evaluated
                 *   for each e ∈ neighbourhood(c) at layer lc // update C and W
                 *     if e ∉ v
                 *       v ← v ⋃ e
                 *       f ← get furthest element from W to q
                 *       if distance(e, q) < distance(f, q) or │W│ < ef
                 *         C ← C ⋃ e
                 *         W ← W ⋃ e
                 *         if │W│ > ef
                 *           remove furthest element from W to q
                 * return W
                 */
                var topCandidates = new BinaryHeap<ValueTuple<TDistance, int>>(new List<(TDistance, int)>(k), Core.FartherIsOnTop);
                var candidates = new BinaryHeap<ValueTuple<TDistance, int>>(ExpansionBuffer, Core.CloserIsOnTop);

                var entry = (travelingCosts.From(entryPointId), entryPointId);
                (var farthestResultDist, var farthestResultId) = entry;

                topCandidates.Push(entry);
                candidates.Push(entry);
                VisitedSet.Add(entryPointId);

                try
                {
                    // run bfs
                    while (candidates.Buffer.Count > 0)
                    {
                        // get next candidate to check and expand
                        (var closestCandidateDist, var closestCandidateId) = candidates.Buffer[0];
                        if (closestCandidateDist > farthestResultDist && topCandidates.Count >= k)
                        {
                            break;
                        }
                        candidates.Pop(); // Delay heap reordering in case of early break 

                        // expand candidate
                        var neighboursIds = Core.Nodes[closestCandidateId][layer];

                        for (int i = 0; i < neighboursIds.Count; ++i)
                        {
                            int neighbourId = neighboursIds[i];
                            if (VisitedSet.Contains(neighbourId)) continue;

                            var neighbourDistance = travelingCosts.From(neighbourId);

                            // enqueue perspective neighbours to expansion list
                            if (topCandidates.Buffer.Count < k || neighbourDistance < farthestResultDist)
                            {
                                candidates.Push((neighbourDistance, neighbourId));
                                topCandidates.Push((neighbourDistance, neighbourId));

                                if (topCandidates.Buffer.Count > 0)
                                    (farthestResultDist, farthestResultId) = topCandidates.Buffer[0];
                            }

                            // update visited list
                            VisitedSet.Add(neighbourId);
                        }
                    }

                    ExpansionBuffer.Clear();
                    VisitedSet.Clear();

                    return topCandidates.Buffer;
                }
                catch
                {
                    //Throws if the collection changed, otherwise propagates the original exception
                    GraphChangedException.ThrowIfChanged(ref version, versionAtStart);
                    throw;
                }
            }
        }
    }
}
