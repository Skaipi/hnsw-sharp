// <copyright file="Graph.Core.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Concurrent;
    using System.Collections.Generic;
    using System.IO;
    using System.Numerics;
    using System.Runtime.CompilerServices;
    using System.Threading;
    using MessagePack;

    using static HNSW.Net.EventSources;

    internal partial class Graph<TItem, TDistance> where TDistance : struct, IFloatingPoint<TDistance>
    {
        internal class Core
        {
            private readonly Func<TItem, TItem, TDistance> Distance;

            internal List<object> NodesLocks { get; private set; }

            internal List<Node> Nodes { get; private set; }

            internal object ItemsReadLock { get; private set; }

            internal List<object> ItemsWriteLocks { get; private set; }

            internal ConcurrentDictionary<int, TItem> Items { get; private set; }

            internal HashSet<int> RemovedIndexes { get; private set; }

            internal Algorithms.Algorithm<TItem, TDistance> Algorithm { get; private set; }

            internal SmallWorld<TItem, TDistance>.Parameters Parameters { get; private set; }
            internal IComparer<NodeDistance<TDistance>> FartherIsOnTop;
            internal IComparer<NodeDistance<TDistance>> CloserIsOnTop;

            internal Core(Func<TItem, TItem, TDistance> distance, SmallWorld<TItem, TDistance>.Parameters parameters)
            {
                Distance = distance;
                Parameters = parameters;

                var initialSize = Math.Max(1024, parameters.InitialItemsSize);

                ItemsReadLock = new object();
                NodesLocks = new List<object>(initialSize);
                ItemsWriteLocks = new List<object>(65535); // 2^16 - 1

                RemovedIndexes = new HashSet<int>();
                Nodes = new List<Node>(initialSize);
                Items = new ConcurrentDictionary<int, TItem>(65536, initialSize); // 2^16 amount of locks

                switch (Parameters.NeighbourHeuristic)
                {
                    case NeighbourSelectionHeuristic.SelectSimple:
                        {
                            Algorithm = new Algorithms.Algorithm3<TItem, TDistance>(this);
                            break;
                        }
                    case NeighbourSelectionHeuristic.SelectHeuristic:
                        {
                            Algorithm = new Algorithms.Algorithm4<TItem, TDistance>(this);
                            break;
                        }
                    case NeighbourSelectionHeuristic.CustomHeuristic:
                        {
                            Algorithm = new Algorithms.Algorithm5<TItem, TDistance>(this);
                            break;
                        }
                }

                FartherIsOnTop = new DistanceComparer<TDistance>();
                CloserIsOnTop = new ReverseDistanceComparer<TDistance>();
            }

            internal IReadOnlyList<int> AddItems(IReadOnlyList<TItem> items, IProvideRandomValues generator)
            {
                int newCount = items.Count;
                var newIDs = new List<int>();
                int index = 0;
                foreach (int vacantId in RemovedIndexes)
                {
                    Nodes[vacantId] = Algorithm.NewNode(vacantId, RandomLayer(generator, Parameters.LevelLambda));
                    Items.TryAdd(vacantId, items[index]);
                    newIDs.Add(vacantId);
                    RemovedIndexes.Remove(vacantId);
                    index++;
                }

                int id0 = Nodes.Count;
                for (int id = 0; id < newCount; ++id)
                {
                    var newId = id0 + id;
                    NodesLocks.Add(new object());
                    Nodes.Add(Algorithm.NewNode(newId, RandomLayer(generator, Parameters.LevelLambda)));
                    Items.TryAdd(newId, items[id + index]);
                    newIDs.Add(newId);
                }
                return newIDs;
            }

            internal void Serialize(Stream stream)
            {
                MessagePackSerializer.Serialize(stream, Nodes);
            }

            internal void Deserialize(IReadOnlyList<TItem> items, Stream stream)
            {
                // readStrict: true -> removed, as not available anymore on MessagePack 2.0 - also probably not necessary anymore
                //                     see https://github.com/neuecc/MessagePack-CSharp/pull/663
                Nodes = MessagePackSerializer.Deserialize<List<Node>>(stream);

                int index = 0;
                foreach (var node in Nodes)
                {
                    Items.TryAdd(node.Id, items[index]);
                    index++;
                }
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            internal TDistance GetDistance(int fromId, int toId)
            {
                return Distance(Items[fromId], Items[toId]);
            }

            private static int RandomLayer(IProvideRandomValues generator, double lambda)
            {
                var r = -Math.Log(generator.NextFloat()) * lambda;
                return (int)r;
            }
        }
    }
}
