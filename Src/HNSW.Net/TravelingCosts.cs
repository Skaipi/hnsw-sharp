// <copyright file="TravelingCosts.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;

    /// <summary>
    /// Implementation of distance calculation from an arbitrary point to the given destination.
    /// </summary>
    /// <typeparam name="TItem">Type of the points.</typeparam>
    /// <typeparam name="TDistance">Type of the distance.</typeparam>
    public class TravelingCosts<TItem, TDistance>
    {
        // private static readonly Comparer<TDistance> DistanceComparer = Comparer<TDistance>.Default;

        private readonly Func<TItem, TItem, TDistance> Distance;

        public TravelingCosts(Func<TItem, TItem, TDistance> distance, TItem destination)
        {
            Distance = distance;
            Destination = destination;
        }

        public TItem Destination { get; }

        public TDistance From(TItem departure)
        {
            return Distance(departure, Destination);
        }
    }
}
