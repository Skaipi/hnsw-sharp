namespace HNSW.Net
{
    using System.Collections.Generic;
    using System.Numerics;
    using System.Runtime.CompilerServices;

    sealed class DistanceComparer<T> : IComparer<(T, int)> where T : IFloatingPoint<T>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Compare((T, int) x, (T, int) y)
        {
            if (x.Item1 < y.Item1) return -1;
            if (x.Item1 > y.Item1) return 1;
            return x.Item1.CompareTo(y.Item1);
        }
    }

    sealed class ReverseDistanceComparer<T> : IComparer<(T, int)> where T : IFloatingPoint<T>
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int Compare((T, int) x, (T, int) y)
        {
            if (x.Item1 > y.Item1) return -1;
            if (x.Item1 < y.Item1) return 1;
            return y.Item1.CompareTo(x.Item1);
        }
    }
}