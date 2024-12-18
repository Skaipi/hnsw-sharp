// <copyright file="Node.Algorithm3.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;

    internal partial class Algorithms
    {
        /// <summary>
        /// The implementation of the SELECT-NEIGHBORS-SIMPLE(q, C, M) algorithm.
        /// Article: Section 4. Algorithm 3.
        /// </summary>
        /// <typeparam name="TItem">The typeof the items in the small world.</typeparam>
        /// <typeparam name="TDistance">The type of the distance in the small world.</typeparam>
        internal class Algorithm3<TItem, TDistance> : Algorithm<TItem, TDistance> where TDistance : struct, IComparable<TDistance>
        {
            public Algorithm3(Graph<TItem, TDistance>.Core graphCore) : base(graphCore)
            {
            }

            /// <inheritdoc/>
            internal override List<int> SelectBestForConnecting(List<ValueTuple<TDistance, int>> candidatesIds, TravelingCosts<int, TDistance> travelingCosts, int layer)
            {
                /*
                 * q ← this
                 * return M nearest elements from C to q
                 */


                // !NO COPY! in-place selection
                var bestN = GetM(layer);

                var fartherIsOnTop = Comparer<(TDistance Distance, int Index)>.Create((x, y) => x.Distance.CompareTo(y.Distance));

                var candidatesHeap = new BinaryHeap<ValueTuple<TDistance, int>>(candidatesIds, fartherIsOnTop);
                while (candidatesHeap.Buffer.Count > bestN)
                {
                    var discardedCandidate = candidatesHeap.Pop();
                }

                return candidatesHeap.Buffer.ConvertAll(x => x.Item2);
            }
        }
    }
}