using System;
using System.Collections.Generic;
using System.IO;
using NUnit.Framework;

namespace MicroBVH.Tests
{
	/// <summary>
	/// Validates the indexed-geometry build path against the indexed sections of the reference dump
	/// (Tools/RefDump/refdump.cpp): a BVH built over a welded vertex array plus a separate index
	/// array, where every layout and traversal addresses vertices through Bvh.VertIdx instead of
	/// prim * 3. The welded mesh comes out of the dump rather than being re-welded here, so the two
	/// sides cannot drift apart over the welding rule.
	/// </summary>
	public class BvhIndexedTests
	{
		/// <summary>Barycentric distance to a triangle edge below which a hit counts as grazing.</summary>
		const float GrazingEps = 1e-3f;

		const float RelativeTolerance = 1e-4f;

		/// <summary>Number of differing entries listed in an assertion message before it is truncated.</summary>
		const int MaxReportedDiffs = 4;

		/// <summary>The welded mesh and the reference sections that go with it, plus the ray set used to trace it.</summary>
		private class IndexedScene
		{
			public RefDumpFile Ref;
			public uint TriCount;
			public BvhVec4[] Vertices;
			public uint[] Indices;
			public BvhVec3[] Origins;
			public BvhVec3[] Directions;
			public Intersection[] Hits;
			public int[] Occluded;

			public int RayCount => Origins.Length;
		}

		/// <summary>
		/// Loads the indexed sections of a scene's reference dump. Ignores the test when the dump is
		/// missing.
		/// </summary>
		static IndexedScene LoadScene( string sceneName )
		{
			string refPath = BvhSceneFile.TestDataPath( sceneName + ".ref" );
			if ( !File.Exists( refPath ) )
			{
				Assert.Ignore( $"missing {refPath}; see the README for how to generate the reference dumps" );
			}
			RefDumpFile refFile = RefDumpFile.Load( refPath );

			IndexedScene scene = new IndexedScene();
			scene.Ref = refFile;
			scene.TriCount = refFile.TriCount;
			scene.Vertices = refFile.WeldedVertices;
			scene.Indices = refFile.Indices;
			int rayCount = refFile.IndexedRays.Length;
			scene.Origins = new BvhVec3[ rayCount ];
			scene.Directions = new BvhVec3[ rayCount ];
			scene.Hits = new Intersection[ rayCount ];
			scene.Occluded = new int[ rayCount ];
			for ( int i = 0; i < rayCount; i++ )
			{
				scene.Origins[ i ] = refFile.IndexedRays[ i ].O;
				scene.Directions[ i ] = refFile.IndexedRays[ i ].D;
			}
			return scene;
		}

		/// <summary>Traces the whole ray set of a scene through any layout, replacing the Unity test's Burst job.</summary>
		static void TraceScene( Bvh layout, IndexedScene scene )
		{
			for ( int i = 0; i < scene.RayCount; i++ )
			{
				Ray ray = new Ray( scene.Origins[ i ], scene.Directions[ i ] );
				layout.Intersect( ref ray );
				scene.Hits[ i ] = ray.Hit;
				scene.Occluded[ i ] = layout.IsOccluded( new Ray( scene.Origins[ i ], scene.Directions[ i ] ) ) ? 1 : 0;
			}
		}

		static bool BitsEqual( float a, float b )
		{
			return BvhMath.AsUint( a ) == BvhMath.AsUint( b );
		}

		static bool BitsEqual( BvhVec3 a, BvhVec3 b )
		{
			return BitsEqual( a.x, b.x ) && BitsEqual( a.y, b.y ) && BitsEqual( a.z, b.z );
		}

		static bool NodesEqual( BvhNode a, BvhNode b )
		{
			return BitsEqual( a.AabbMin, b.AabbMin ) && a.LeftFirst == b.LeftFirst
				&& BitsEqual( a.AabbMax, b.AabbMax ) && a.TriCount == b.TriCount;
		}

		static bool IsGrazing( float u, float v )
		{
			return u < GrazingEps || v < GrazingEps || ( 1f - u - v ) < GrazingEps;
		}

		static string Describe( List<int> diffs )
		{
			if ( diffs.Count == 0 )
			{
				return "none";
			}
			string list = string.Join( ", ", diffs.GetRange( 0, Math.Min( diffs.Count, MaxReportedDiffs ) ) );
			return $"{diffs.Count} differing, first at [{list}]";
		}

		static void CompareNodes( string label, BvhNode[] nodes, uint usedNodes, uint expectedUsedNodes, BvhNode[] expected )
		{
			Assert.AreEqual( expectedUsedNodes, usedNodes, label + " UsedNodes" );
			List<int> diffs = new List<int>();
			for ( int i = 0; i < expected.Length; i++ )
			{
				if ( !NodesEqual( nodes[ i ], expected[ i ] ) )
				{
					diffs.Add( i );
				}
			}
			TestContext.WriteLine( $"{label}: node mismatches {diffs.Count}/{expected.Length}" );
			Assert.AreEqual( 0, diffs.Count, $"{label} node mismatch: {Describe( diffs )}" );
		}

		static void ComparePrimIdx( string label, uint[] primIdx, int count, uint[] expected )
		{
			Assert.AreEqual( expected.Length, count, label + " prim count" );
			List<int> diffs = new List<int>();
			for ( int i = 0; i < expected.Length; i++ )
			{
				if ( primIdx[ i ] != expected[ i ] )
				{
					diffs.Add( i );
				}
			}
			Assert.AreEqual( 0, diffs.Count, $"{label} primIdx mismatch: {Describe( diffs )}" );
		}

		/// <summary>
		/// Compares a traced ray set against one of the indexed reference ray sections. Same-distance
		/// hits on a different primitive are legitimate ties between overlapping triangles, and hits
		/// within a hair of a triangle edge are decided by last-bit rounding; both are counted
		/// separately and do not fail the test, exactly as in the other reference suites.
		/// </summary>
		static void CompareRays( string label, RefDumpFile.RayHit[] expected, Intersection[] hits, int[] occluded )
		{
			int rayCount = expected.Length;
			int mismatches = 0, ties = 0, grazing = 0, hitCount = 0, occlusionMismatches = 0;
			for ( int i = 0; i < rayCount; i++ )
			{
				RefDumpFile.RayHit rh = expected[ i ];
				Intersection hit = hits[ i ];
				bool refHit = rh.T < BvhConstants.Far;
				bool gotHit = hit.T < BvhConstants.Far;
				bool sameT = refHit && gotHit
					&& MathF.Abs( hit.T - rh.T ) <= RelativeTolerance * BvhMath.Max( MathF.Abs( rh.T ), 1e-12f );
				if ( refHit != gotHit || ( refHit && !sameT ) )
				{
					if ( ( refHit && IsGrazing( rh.U, rh.V ) ) || ( gotHit && IsGrazing( hit.U, hit.V ) ) )
					{
						grazing++;
					}
					else
					{
						if ( mismatches < 3 )
						{
							TestContext.WriteLine( $"  mismatch ray {i}: O {rh.O} D {rh.D} ref t {rh.T} u {rh.U} v {rh.V} prim {rh.Prim} | got t {hit.T} u {hit.U} v {hit.V} prim {hit.Prim}" );
						}
						mismatches++;
					}
				}
				else if ( refHit )
				{
					hitCount++;
					if ( hit.Prim != rh.Prim )
					{
						ties++;
					}
				}
				if ( ( occluded[ i ] != 0 ) != ( rh.OccludedFull != 0 ) )
				{
					occlusionMismatches++;
				}
			}
			float hitRatio = ( float )hitCount / rayCount;
			TestContext.WriteLine( $"{label}: hit ratio {hitRatio:P2}, mismatches {mismatches}/{rayCount}, same-distance ties {ties}, grazing {grazing}, occlusion mismatches {occlusionMismatches}" );
			Assert.AreEqual( 0, mismatches, label + " intersect mismatches" );
			Assert.AreEqual( 0, occlusionMismatches, label + " occlusion mismatches" );
		}

		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void Build_MatchesIndexedReference( string sceneName )
		{
			IndexedScene scene = LoadScene( sceneName );
			Bvh bvh = new Bvh();
			bvh.Build( scene.Vertices, scene.Indices, scene.TriCount );

			Assert.IsTrue( bvh.VertIdx != null, "BvhOverIndices" );
			CompareNodes( $"{sceneName} indexed", bvh.Nodes, bvh.UsedNodes, scene.Ref.IndexedUsedNodes, scene.Ref.IndexedNodes );
			ComparePrimIdx( $"{sceneName} indexed", bvh.PrimIdx, ( int )bvh.TriCount, scene.Ref.IndexedPrimIdx );

			TraceScene( bvh, scene );
			CompareRays( $"{sceneName} indexed Bvh", scene.Ref.IndexedRays, scene.Hits, scene.Occluded );
		}
	}
}
