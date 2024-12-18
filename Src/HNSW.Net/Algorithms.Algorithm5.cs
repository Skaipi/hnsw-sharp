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

            internal override List<int> SelectBestForConnecting(List<ValueTuple<TDistance, int>> candidatesIds, TravelingCosts<int, TDistance> travelingCosts, int layer)
            {
                var layerM = GetM(layer);
                if (candidatesIds.Count < layerM)
                {
                    return candidatesIds.ConvertAll(x => x.Item2);
                }

                var fartherIsOnTop = Comparer<ValueTuple<TDistance, int>>.Create((x, y) => x.Item1.CompareTo(y.Item1));
                var closerIsOnTop = Comparer<ValueTuple<TDistance, int>>.Create((x, y) => -x.Item1.CompareTo(y.Item1));

                var resultList = new List<ValueTuple<TDistance, int>>(layerM + 1);
                var candidatesHeap = new BinaryHeap<ValueTuple<TDistance, int>>(candidatesIds, closerIsOnTop);

                while (candidatesHeap.Count > 0)
                {
                    if (resultList.Count >= layerM)
                        break;

                    var currentCandidate = candidatesHeap.Pop();
                    var candidateDist = currentCandidate.Item1;

                    if (resultList.All(connectedNode => DistanceUtils.GreaterThan(connectedNode.Item1, candidateDist)))
                    {
                        resultList.Add(currentCandidate);
                    }
                }

                return resultList.ConvertAll(x => x.Item2);
            }
        }
    }
}