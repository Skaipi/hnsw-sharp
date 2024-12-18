using System;
using System.Numerics;
using System.Runtime.CompilerServices;

namespace HNSW.Net
{
  /// <summary>
  /// Calculates Euclidean distance between two vectors.
  /// </summary>
  public static class EuclideanDistance
  {
    /// <summary>
    /// Calculates Euclidean distance without any SIMD optimizations.
    /// </summary>
    /// <param name="u">Left vector.</param>
    /// <param name="v">Right vector.</param>
    /// <returns>The Euclidean distance between u and v.</returns>
    public static float NonOptimized(float[] u, float[] v)
    {
      if (u.Length != v.Length)
      {
        throw new ArgumentException("Vectors have non-matching dimensions");
      }

      float sum = 0.0f;
      for (int i = 0; i < u.Length; ++i)
      {
        float diff = u[i] - v[i];
        sum += diff * diff;
      }

      return sum;
    }

    /// <summary>
    /// Calculates Euclidean distance optimized using SIMD instructions if available.
    /// </summary>
    /// <param name="u">Left vector.</param>
    /// <param name="v">Right vector.</param>
    /// <returns>The Euclidean distance between u and v.</returns>
    public static float SIMD(float[] u, float[] v)
    {
      if (!Vector.IsHardwareAccelerated)
      {
        throw new NotSupportedException($"SIMD version of {nameof(EuclideanDistance)} is not supported");
      }

      if (u.Length != v.Length)
      {
        throw new ArgumentException("Vectors have non-matching dimensions");
      }

      int length = u.Length;
      int step = Vector<float>.Count;

      float sum = 0f;
      int i, to = length - step;
      // Use a vector accumulator to reduce floating-point summation error
      Vector<float> accumulator = Vector<float>.Zero;

      for (i = 0; i <= to; i += step)
      {
        var vu = new Vector<float>(u, i);
        var vv = new Vector<float>(v, i);
        var diff = vu - vv;
        accumulator += diff * diff;
      }

      // Sum up the vector accumulator
      for (int j = 0; j < step; j++)
      {
        sum += accumulator[j];
      }

      // Handle remainder elements
      for (; i < length; ++i)
      {
        float diff = u[i] - v[i];
        sum += diff * diff;
      }

      return sum;
    }

    /// <summary>
    /// A fully unrolled SIMD dot-like pattern for scenarios where the dimension is large and 
    /// we want to minimize loop overhead. Useful if you have very large vectors and need maximum speed.
    /// </summary>
    /// <param name="u">Left vector.</param>
    /// <param name="v">Right vector.</param>
    /// <returns>Euclidean distance between u and v.</returns>
    public static float SIMDUnrolled(float[] u, float[] v)
    {
      if (!Vector.IsHardwareAccelerated)
      {
        throw new NotSupportedException($"SIMD version of {nameof(EuclideanDistance)} is not supported");
      }

      if (u.Length != v.Length)
      {
        throw new ArgumentException("Vectors have non-matching dimensions");
      }

      float sum = 0f;
      int count = u.Length;
      int offset = 0;

      // Pre-calculate steps
      int vs1 = Vector<float>.Count;
      int vs2 = 2 * vs1;
      int vs3 = 3 * vs1;
      int vs4 = 4 * vs1;

      // Vector accumulator
      Vector<float> accumulator = Vector<float>.Zero;

      while (count >= vs4)
      {
        accumulator += DistSquaredVector(new Vector<float>(u, offset), new Vector<float>(v, offset));
        accumulator += DistSquaredVector(new Vector<float>(u, offset + vs1), new Vector<float>(v, offset + vs1));
        accumulator += DistSquaredVector(new Vector<float>(u, offset + vs2), new Vector<float>(v, offset + vs2));
        accumulator += DistSquaredVector(new Vector<float>(u, offset + vs3), new Vector<float>(v, offset + vs3));

        count -= vs4;
        offset += vs4;
      }

      if (count >= vs2)
      {
        accumulator += DistSquaredVector(new Vector<float>(u, offset), new Vector<float>(v, offset));
        accumulator += DistSquaredVector(new Vector<float>(u, offset + vs1), new Vector<float>(v, offset + vs1));
        count -= vs2;
        offset += vs2;
      }

      if (count >= vs1)
      {
        accumulator += DistSquaredVector(new Vector<float>(u, offset), new Vector<float>(v, offset));
        count -= vs1;
        offset += vs1;
      }

      // Sum partial results from accumulator
      for (int i = 0; i < Vector<float>.Count; i++)
      {
        sum += accumulator[i];
      }

      // Handle remainder
      while (count > 0)
      {
        float diff = u[offset] - v[offset];
        sum += diff * diff;
        offset++;
        count--;
      }

      return sum;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector<float> DistSquaredVector(Vector<float> va, Vector<float> vb)
    {
      var diff = va - vb;
      return diff * diff;
    }
  }
}

