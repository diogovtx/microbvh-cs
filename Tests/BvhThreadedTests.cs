using System;
using System.Diagnostics;
using System.IO;
using NUnit.Framework;

namespace MicroBVH.Tests
{
	/// <summary>
	/// Tests for Bvh.UseThreadedBuild. A threaded build hands out node indices from an atomic counter,
	/// so its node numbering is not reproducible and cannot be compared against the C++ dump; what is
	/// reproducible is the tree shape, so the threaded trees are compared against the serial ones by
	/// walking both in the same depth-first order. The reference ray sets are traced through the
	/// threaded trees as well, and a build below BvhConstants.MtBuildThreshold is checked to be the
	/// serial build, byte for byte.
	/// </summary>
	public class BvhThreadedTests
	{
		/// <summary>cryteksponza has 262267 primitives, well above the threading threshold.</summary>
		const string LargeScene = "cryteksponza";
		/// <summary>suzanne has 15488 primitives, below the threading threshold.</summary>
		const string SmallScene = "suzanne";
		const float MaxMismatchFraction = 0.0005f;

		static bool TryGetPaths( string sceneName, out string binPath, out string refPath )
		{
			binPath = BvhSceneFile.TestDataPath( sceneName + ".bin" );
			refPath = BvhSceneFile.TestDataPath( sceneName + ".ref" );
			return File.Exists( binPath ) && File.Exists( refPath );
		}

		static bool BitsEqual( float a, float b )
		{
			return BvhMath.AsUint( a ) == BvhMath.AsUint( b );
		}

		static bool BoundsEqual( BvhNode a, BvhNode b )
		{
			return BitsEqual( a.AabbMin.x, b.AabbMin.x )
				&& BitsEqual( a.AabbMin.y, b.AabbMin.y )
				&& BitsEqual( a.AabbMin.z, b.AabbMin.z )
				&& BitsEqual( a.AabbMax.x, b.AabbMax.x )
				&& BitsEqual( a.AabbMax.y, b.AabbMax.y )
				&& BitsEqual( a.AabbMax.z, b.AabbMax.z );
		}

		static Bvh BuildScene( BvhVec4[] verts, uint triCount, bool threaded )
		{
			Bvh bvh = new Bvh();
			bvh.UseThreadedBuild = threaded;
			bvh.Build( verts, triCount );
			return bvh;
		}

		/// <summary>Compares the primitive index sets of two matching leaves, order-independently.</summary>
		static void AssertSameLeafPrims( Bvh serial, BvhNode a, Bvh threaded, BvhNode b, uint nodeA, string label )
		{
			int count = ( int )a.TriCount;
			uint[] primsA = new uint[ count ];
			uint[] primsB = new uint[ count ];
			for ( int i = 0; i < count; i++ )
			{
				primsA[ i ] = serial.PrimIdx[ a.LeftFirst + i ];
				primsB[ i ] = threaded.PrimIdx[ b.LeftFirst + i ];
			}
			Array.Sort( primsA );
			Array.Sort( primsB );
			for ( int i = 0; i < count; i++ )
			{
				Assert.AreEqual( primsA[ i ], primsB[ i ], $"{label}: leaf primitive sets differ at serial node {nodeA}, entry {i}" );
			}
		}

		/// <summary>
		/// Walks both trees from the root in the same depth-first order and asserts that every node
		/// pair has bit-identical bounds, the same leaf/interior kind and, for leaves, the same
		/// multiset of primitive indices.
		/// </summary>
		static void AssertSameTree( Bvh serial, Bvh threaded, string label )
		{
			const int MaxDepth = 256;
			Span<uint> stackA = stackalloc uint[ MaxDepth ];
			Span<uint> stackB = stackalloc uint[ MaxDepth ];
			int stackPtr = 0;
			uint a = 0, b = 0;
			int nodes = 0, leaves = 0, prims = 0;
			while ( true )
			{
				BvhNode na = serial.Nodes[ a ];
				BvhNode nb = threaded.Nodes[ b ];
				nodes++;
				Assert.IsTrue( BoundsEqual( na, nb ), $"{label}: bounds differ at serial node {a} / threaded node {b}" );
				Assert.AreEqual( na.IsLeaf, nb.IsLeaf, $"{label}: leaf/interior differs at serial node {a} / threaded node {b}" );
				if ( na.IsLeaf )
				{
					Assert.AreEqual( na.TriCount, nb.TriCount, $"{label}: leaf size differs at serial node {a} / threaded node {b}" );
					AssertSameLeafPrims( serial, na, threaded, nb, a, label );
					leaves++;
					prims += ( int )na.TriCount;
					if ( stackPtr == 0 )
					{
						break;
					}
					stackPtr--;
					a = stackA[ stackPtr ];
					b = stackB[ stackPtr ];
				}
				else
				{
					Assert.Less( stackPtr, MaxDepth, $"{label}: tree deeper than the walk stack" );
					stackA[ stackPtr ] = na.LeftFirst + 1;
					stackB[ stackPtr ] = nb.LeftFirst + 1;
					stackPtr++;
					a = na.LeftFirst;
					b = nb.LeftFirst;
				}
			}
			TestContext.WriteLine( $"{label}: {nodes} nodes walked, {leaves} leaves, {prims} leaf prims, bounds bit-identical, leaf prim sets equal" );
		}

		[Test]
		public void ThreadedBuild_MatchesSerialTree()
		{
			if ( !TryGetPaths( LargeScene, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; see the README for how to get the scenes and generate the reference dumps" );
			}

			BvhVec4[] verts = BvhSceneFile.Load( binPath, out uint triCount );
			Bvh serial = BuildScene( verts, triCount, false );
			Bvh threaded = BuildScene( verts, triCount, true );

			string label = LargeScene;
			TestContext.WriteLine( $"{label}: serial nodes {serial.UsedNodes}, threaded nodes {threaded.UsedNodes}, subtree jobs {threaded.ThreadedSubtrees}" );
			Assert.AreEqual( 0u, serial.ThreadedSubtrees, "the serial build must not report subtree jobs" );
			Assert.Greater( threaded.ThreadedSubtrees, 1u, "the threaded build must have run more than one subtree job" );
			Assert.AreEqual( serial.UsedNodes, threaded.UsedNodes, "UsedNodes" );
			AssertSameTree( serial, threaded, label );
		}

		[Test]
		public void ThreadedBuild_BinnedIntersectMatchesReference()
		{
			if ( !TryGetPaths( LargeScene, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; see the README for how to get the scenes and generate the reference dumps" );
			}

			BvhVec4[] verts = BvhSceneFile.Load( binPath, out uint triCount );
			Bvh bvh = BuildScene( verts, triCount, true );
			RefDumpFile refFile = RefDumpFile.Load( refPath );
			int mismatches = 0, ties = 0, hitCount = 0;

			for ( int i = 0; i < refFile.Rays.Length; i++ )
			{
				RefDumpFile.RayHit rh = refFile.Rays[ i ];
				Ray ray = new Ray( rh.O, rh.D );
				bvh.Intersect( ref ray );

				bool refHit = rh.T < BvhConstants.Far;
				bool gotHit = ray.Hit.T < BvhConstants.Far;
				if ( refHit != gotHit )
				{
					mismatches++;
					continue;
				}
				if ( refHit )
				{
					hitCount++;
					bool sameT = MathF.Abs( ray.Hit.T - rh.T ) <= 1e-4f * BvhMath.Max( MathF.Abs( rh.T ), 1e-12f );
					if ( !sameT )
					{
						mismatches++;
					}
					else if ( ray.Hit.Prim != rh.Prim )
					{
						// A different primitive at the same distance is a legitimate tie between
						// overlapping triangles, decided by traversal order and last-bit rounding.
						ties++;
					}
				}
			}

			TestContext.WriteLine( $"{LargeScene} binned threaded rays: hits {hitCount}/{refFile.Rays.Length}, mismatches {mismatches}, same-distance ties {ties}" );
			Assert.LessOrEqual( mismatches, ( int )( refFile.Rays.Length * MaxMismatchFraction ), "mismatch rate too high" );
		}

		/// <summary>
		/// suzanne is below BvhConstants.MtBuildThreshold, so setting the flag must change nothing:
		/// the build stays serial and produces the very same node and index arrays.
		/// </summary>
		[Test]
		public void BelowThreshold_ThreadedFlagBuildsTheSerialTree()
		{
			if ( !TryGetPaths( SmallScene, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; see the README for how to get the scenes and generate the reference dumps" );
			}

			BvhVec4[] verts = BvhSceneFile.Load( binPath, out uint triCount );
			Assert.Less( triCount, BvhConstants.MtBuildThreshold, "this test needs a scene below the threading threshold" );
			Bvh serial = BuildScene( verts, triCount, false );
			Bvh flagged = BuildScene( verts, triCount, true );

			Assert.AreEqual( 0u, flagged.ThreadedSubtrees, "a build below the threshold must not run subtree jobs" );
			Assert.AreEqual( serial.UsedNodes, flagged.UsedNodes, "UsedNodes" );
			int nodeMismatches = 0;
			for ( uint i = 0; i < serial.UsedNodes; i++ )
			{
				BvhNode a = serial.Nodes[ i ], b = flagged.Nodes[ i ];
				if ( !BoundsEqual( a, b ) || a.LeftFirst != b.LeftFirst || a.TriCount != b.TriCount )
				{
					nodeMismatches++;
				}
			}
			int idxMismatches = 0;
			int idxCount = ( int )serial.TriCount;
			for ( int i = 0; i < idxCount; i++ )
			{
				if ( serial.PrimIdx[ i ] != flagged.PrimIdx[ i ] )
				{
					idxMismatches++;
				}
			}
			TestContext.WriteLine( $"{SmallScene} below threshold: {serial.UsedNodes} nodes, node mismatches {nodeMismatches}, primIdx mismatches {idxMismatches}/{idxCount}" );
			Assert.AreEqual( 0, nodeMismatches, "the threaded flag must not change a build below the threshold" );
			Assert.AreEqual( 0, idxMismatches, "the threaded flag must not change the index array below the threshold" );
		}

		/// <summary>Informational: serial versus threaded build time. No assertion.</summary>
		[Test]
		public void BuildTime_SerialVersusThreaded()
		{
			if ( !TryGetPaths( LargeScene, out string binPath, out string refPath ) )
			{
				Assert.Ignore( $"missing {binPath} or {refPath}; see the README for how to get the scenes and generate the reference dumps" );
			}

			BvhVec4[] verts = BvhSceneFile.Load( binPath, out uint triCount );
			Bvh bvh = new Bvh();
			bvh.UseThreadedBuild = false;
			Stopwatch timer = Stopwatch.StartNew();
			bvh.Build( verts, triCount );
			double serialMs = timer.Elapsed.TotalMilliseconds;
			bvh.UseThreadedBuild = true;
			timer.Restart();
			bvh.Build( verts, triCount );
			double threadedMs = timer.Elapsed.TotalMilliseconds;
			TestContext.WriteLine( $"{LargeScene} build: serial {serialMs:0.0} ms, threaded {threadedMs:0.0} ms over {bvh.ThreadedSubtrees} subtrees ({serialMs / threadedMs:0.00}x)" );
		}
	}
}
