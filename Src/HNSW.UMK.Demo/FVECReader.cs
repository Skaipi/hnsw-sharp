using System;

namespace HNSW.UMK.Demo;

public static class FVECReader
{
  public static List<float[]> ReadFVEC(string fileName, int maxVectors = int.MaxValue)
  {
    var vectors = new List<float[]>();
    var buffSize = 16777216; // 2^24

    using (var fs = new FileStream(fileName, FileMode.Open, FileAccess.Read, FileShare.Read, bufferSize: buffSize, useAsync: false))
    {
      byte[] dimBuffer = new byte[4];
      while (fs.Position < fs.Length)
      {
        // Read dimension (4 bytes)
        if (fs.Read(dimBuffer, 0, 4) < 4)
          throw new Exception("Innvalid FVEC file dim header!");

        int dim = BitConverter.ToInt32(dimBuffer, 0);

        // Read the vector data as bytes
        int floatCount = dim;
        int floatBytes = floatCount * sizeof(float);
        byte[] floatBuffer = new byte[floatBytes];

        int readBytes = fs.Read(floatBuffer, 0, floatBytes);
        if (readBytes < floatBytes)
          throw new Exception("Innvalid FVEC file data!");

        float[] vector = new float[dim];
        Buffer.BlockCopy(floatBuffer, 0, vector, 0, floatBytes);

        vectors.Add(vector);
      }
    }

    if (vectors.Count > maxVectors)
      vectors = vectors.Take(maxVectors).ToList();

    return vectors;
  }
}
