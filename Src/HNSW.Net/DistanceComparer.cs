namespace HNSW.Net
{
  using System;
  using System.Collections.Generic;
  using System.Numerics;

  sealed class DistanceComparer<T> : IComparer<(T, int)> where T : IFloatingPoint<T>
  {
    public int Compare((T, int) x, (T, int) y)
    {
      return x.Item1.CompareTo(y.Item1);
    }
  }

  sealed class ReverseDistanceComparer<T> : IComparer<(T, int)> where T : IFloatingPoint<T>
  {
    public int Compare((T, int) x, (T, int) y)
    {
      // Simply invert the order by swapping x and y in the CompareTo call
      return y.Item1.CompareTo(x.Item1);
    }
  }
}