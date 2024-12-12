namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    internal partial class Algorithms
    {
        internal sealed class Algorithm5<TItem, TDistance> : Algorithm<TItem, TDistance> where TDistance : struct, IComparable<TDistance>
        {
            public Algorithm5(Graph<TItem, TDistance>.Core graphCore) : base(graphCore)
            {
            }

            internal override (List<int>, BinaryHeap<int>) SelectBestForConnecting(List<int> candidatesIds, TravelingCosts<int, TDistance> travelingCosts, int layer)
            {
                var layerM = GetM(layer);
                if (candidatesIds.Count < layerM)
                {
                    return (candidatesIds, new BinaryHeap<int>());
                }

                IComparer<int> fartherIsOnTop = travelingCosts;
                IComparer<int> closerIsOnTop = fartherIsOnTop.Reverse();

                var resultList = new List<int>(layerM + 1);
                var candidatesHeap = new BinaryHeap<int>(candidatesIds, closerIsOnTop);

                while (candidatesHeap.Count > 0)
                {
                    if (resultList.Count >= layerM)
                        break;

                    var currentCandidate = candidatesHeap.Pop();
                    var candidateDist = travelingCosts.From(currentCandidate);

                    if (resultList.All(connectedNode => DistanceUtils.GreaterThan(travelingCosts.From(connectedNode), candidateDist)))
                    {
                        resultList.Add(currentCandidate);
                    }
                }

                return (resultList, new BinaryHeap<int>());
            }
        }
    }
}
