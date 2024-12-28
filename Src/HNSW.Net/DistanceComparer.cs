namespace HNSW.Net
{
    using System.Collections.Generic;
    using System.Numerics;
    using System.Runtime.CompilerServices;

    sealed class DistanceComparer<T> : IComparer<NodeDistance<T>> where T : struct, IFloatingPoint<T>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Compare(NodeDistance<T> x, NodeDistance<T> y)
        {
            if (x.Dist < y.Dist) return -1;
            if (x.Dist > y.Dist) return 1;
            return x.Dist.CompareTo(y.Dist);
        }
    }

    sealed class ReverseDistanceComparer<T> : IComparer<NodeDistance<T>> where T : struct, IFloatingPoint<T>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Compare(NodeDistance<T> x, NodeDistance<T> y)
        {
            if (x.Dist > y.Dist) return -1;
            if (x.Dist < y.Dist) return 1;
            return y.Dist.CompareTo(x.Dist);
        }
    }
}