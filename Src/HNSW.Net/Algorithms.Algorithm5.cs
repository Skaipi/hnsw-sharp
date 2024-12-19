namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Numerics;

    internal partial class Algorithms
    {
        internal sealed class Algorithm5<TItem, TDistance> : Algorithm<TItem, TDistance> where TDistance : struct, IFloatingPoint<TDistance>
        {
            public Algorithm5(Graph<TItem, TDistance>.Core graphCore) : base(graphCore)
            {
            }

            internal override List<int> SelectBestForConnecting(List<ValueTuple<TDistance, int>> candidatesIds, TravelingCosts<int, TDistance> travelingCosts, int layer)
            {
                var layerM = GetM(layer);
                if (candidatesIds.Count < layerM)
                {
                    return candidatesIds.ConvertAll(x => x.Item2);
                }

                Func<int, int, TDistance> distance = GraphCore.GetDistance;
                var resultList = new List<ValueTuple<TDistance, int>>(layerM + 1);
                var candidatesHeap = new BinaryHeap<ValueTuple<TDistance, int>>(candidatesIds, GraphCore.CloserIsOnTop);

                while (candidatesHeap.Count > 0)
                {
                    if (resultList.Count >= layerM)
                        break;

                    var currentCandidate = candidatesHeap.Pop();
                    var candidateDist = currentCandidate.Item1;

                    // Candidate is closer to designated point than any other already connected point (distance to )
                    if (resultList.TrueForAll(connectedNode => distance(connectedNode.Item2, currentCandidate.Item2) > candidateDist))
                    {
                        resultList.Add(currentCandidate);
                    }
                }

                return resultList.ConvertAll(x => x.Item2);
            }
        }
    }
}
