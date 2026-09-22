using System;
using NUnit.Framework;

namespace MicroBVH.Tests
{
	/// <summary>
	/// BVHs over custom geometry: a scene of procedurally generated spheres, built from an AABB
	/// array and from a getAabb callback, traversed through the custom intersection callbacks
	/// and checked against a brute-force loop over all spheres.
	/// </summary>
	public class BvhCustomTests
	{
		private const uint SphereCount = 4096;
		private const int RayCount = 4096;
		/// <summary>Ray length used for the occlusion queries, short enough to leave rays unblocked.</summary>
		private const float OccludeDist = 8f;

		private struct Sphere
		{
			public BvhVec3 Center;
			public float Radius;
		}

		/// <summary>
		/// Minimal xorshift32 port of Unity.Mathematics.Random, reproducing just the members this
		/// test uses. Exact bit-compatibility with Unity is not required, since the BVH is always
		/// checked against a brute-force loop driven by the same callbacks; this only has to be a
		/// reasonable pseudo-random generator.
		/// </summary>
		private struct Rng
		{
			private uint state;

			public Rng( uint seed )
			{
				state = seed;
				NextState();
			}

			public static Rng CreateFromIndex( uint index )
			{
				return new Rng( WangHash( index + 62u ) );
			}

			static uint WangHash( uint n )
			{
				n = ( n ^ 61u ) ^ ( n >> 16 );
				n *= 9u;
				n ^= n >> 4;
				n *= 0x27d4eb2du;
				n ^= n >> 15;
				return n;
			}

			uint NextState()
			{
				uint t = state;
				state ^= state << 13;
				state ^= state >> 17;
				state ^= state << 5;
				return t;
			}

			public uint NextUInt( uint max )
			{
				return ( uint )( ( ( ulong )NextState() * max ) >> 32 );
			}

			public uint NextUInt( uint min, uint max )
			{
				return NextUInt( max - min ) + min;
			}

			public ( uint x, uint y, uint z ) NextUInt3( uint max )
			{
				uint x = NextUInt( max );
				uint y = NextUInt( max );
				uint z = NextUInt( max );
				return ( x, y, z );
			}

			public float NextFloat()
			{
				return BvhMath.AsFloat( 0x3f800000u | ( NextState() >> 9 ) ) - 1f;
			}

			public BvhVec3 NextFloat3()
			{
				float x = NextFloat();
				float y = NextFloat();
				float z = NextFloat();
				return new BvhVec3( x, y, z );
			}

			public BvhVec3 NextFloat3( BvhVec3 min, BvhVec3 max )
			{
				return ( NextFloat3() * ( max - min ) ) + min;
			}

			public BvhVec3 NextFloat3Direction()
			{
				float z = ( NextFloat() * 2f ) - 1f;
				float angle = NextFloat() * 2f * MathF.PI;
				float r = MathF.Sqrt( BvhMath.Max( 1f - ( z * z ), 0f ) );
				return new BvhVec3( MathF.Cos( angle ) * r, MathF.Sin( angle ) * r, z );
			}
		}

		/// <summary>
		/// The custom geometry. The spheres are derived from the primitive index alone, so the
		/// callbacks need no scene data and the same code can generate the reference AABBs.
		/// </summary>
		private static class SphereGeometry
		{
			/// <summary>
			/// Center and radius of one sphere. Everything is derived with integer arithmetic and
			/// scaled by a power of two, so no rounding takes place at all.
			/// </summary>
			public static Sphere SphereAt( uint prim )
			{
				Rng rng = Rng.CreateFromIndex( prim );
				( uint x, uint y, uint z ) = rng.NextUInt3( 8192u );
				uint r = rng.NextUInt( 205u, 1229u );
				Sphere s;
				s.Center = ( new BvhVec3( ( int )x, ( int )y, ( int )z ) - new BvhVec3( 4096f ) ) * ( 1f / 1024f );
				s.Radius = r * ( 1f / 4096f );
				return s;
			}

			/// <summary>
			/// Analytic ray/sphere intersection: nearest root in front of the origin, or false when
			/// there is none. The direction is expected to be normalised.
			/// </summary>
			public static bool SphereHit( BvhVec3 o, BvhVec3 d, Sphere s, out float t )
			{
				BvhVec3 oc = o - s.Center;
				float b = BvhMath.Dot( oc, d );
				float c = BvhMath.Dot( oc, oc ) - ( s.Radius * s.Radius );
				float disc = ( b * b ) - c;
				t = 0f;
				if ( disc < 0f )
				{
					return false;
				}
				float sq = MathF.Sqrt( disc );
				t = -b - sq;
				if ( t < 0f )
				{
					t = -b + sq;
				}
				return t > 0f;
			}

			public static void GetAabb( uint prim, out BvhVec3 aabbMin, out BvhVec3 aabbMax )
			{
				Sphere s = SphereAt( prim );
				aabbMin = s.Center - new BvhVec3( s.Radius );
				aabbMax = s.Center + new BvhVec3( s.Radius );
			}

			public static bool Intersect( ref Ray ray, uint prim )
			{
				if ( !SphereHit( ray.O, ray.D, SphereAt( prim ), out float t ) || t >= ray.Hit.T )
				{
					return false;
				}
				ray.Hit.T = t;
				ray.Hit.U = 0f;
				ray.Hit.V = 0f;
				ray.Hit.Prim = prim;
				return true;
			}

			public static bool IsOccluded( in Ray ray, uint prim )
			{
				return SphereHit( ray.O, ray.D, SphereAt( prim ), out float t ) && t < ray.Hit.T;
			}
		}

		/// <summary>
		/// Builds a ray without the Ray constructor, which normalises the direction: doing that
		/// again here could shift a grazing hit by a bit, landing a sphere at a very different
		/// distance. The directions here are already unit length.
		/// </summary>
		private static Ray MakeRay( BvhVec3 o, BvhVec3 d, float t )
		{
			Ray ray = default;
			ray.O = o;
			ray.D = d;
			ray.RD = BvhMath.Rcp( d );
			ray.Mask = BvhConstants.RayMaskIntersectAll;
			ray.Hit.T = t;
			return ray;
		}

		/// <summary>Traverses the BVH, which calls the sphere callbacks for the primitives in a leaf.</summary>
		private static void TraceSpheres( Bvh bvh, BvhVec3[] origins, BvhVec3[] directions, Intersection[] hits, int[] occluded )
		{
			for ( int i = 0; i < origins.Length; i++ )
			{
				Ray ray = MakeRay( origins[ i ], directions[ i ], BvhConstants.Far );
				bvh.Intersect( ref ray );
				hits[ i ] = ray.Hit;
				occluded[ i ] = bvh.IsOccluded( MakeRay( origins[ i ], directions[ i ], OccludeDist ) ) ? 1 : 0;
			}
		}

		/// <summary>
		/// The same queries without a BVH: every sphere is handed to the same callbacks, in order.
		/// Using the callbacks themselves keeps the arithmetic identical to the traversal, so the
		/// results can be compared exactly.
		/// </summary>
		private static void TraceBruteForce( CustomIntersectDelegate intersect, CustomOccludedDelegate isOccluded, BvhVec3[] origins, BvhVec3[] directions, Intersection[] hits, int[] occluded )
		{
			for ( int i = 0; i < origins.Length; i++ )
			{
				Ray ray = MakeRay( origins[ i ], directions[ i ], BvhConstants.Far );
				Ray shadowRay = MakeRay( origins[ i ], directions[ i ], OccludeDist );
				int occ = 0;
				for ( uint p = 0; p < SphereCount; p++ )
				{
					intersect( ref ray, p );
					if ( occ == 0 && isOccluded( shadowRay, p ) )
					{
						occ = 1;
					}
				}
				hits[ i ] = ray.Hit;
				occluded[ i ] = occ;
			}
		}

		private static BvhVec4[] BuildAabbArray()
		{
			BvhVec4[] aabbs = new BvhVec4[ ( int )SphereCount * 2 ];
			for ( uint i = 0; i < SphereCount; i++ )
			{
				Sphere s = SphereGeometry.SphereAt( i );
				aabbs[ ( int )i * 2 ] = new BvhVec4( s.Center - new BvhVec3( s.Radius ), 0f );
				aabbs[ ( ( int )i * 2 ) + 1 ] = new BvhVec4( s.Center + new BvhVec3( s.Radius ), 0f );
			}
			return aabbs;
		}

		private static void MakeRays( BvhVec3[] origins, BvhVec3[] directions )
		{
			Rng rng = new Rng( 0x5EED1234u );
			for ( int i = 0; i < origins.Length; i++ )
			{
				BvhVec3 o = rng.NextFloat3Direction() * 12f;
				origins[ i ] = o;
				directions[ i ] = BvhMath.Normalize( rng.NextFloat3( new BvhVec3( -4f ), new BvhVec3( 4f ) ) - o );
			}
		}

		static bool NodeBitsEqual( BvhNode a, BvhNode b )
		{
			return BvhMath.AsUint( a.AabbMin.x ) == BvhMath.AsUint( b.AabbMin.x )
				&& BvhMath.AsUint( a.AabbMin.y ) == BvhMath.AsUint( b.AabbMin.y )
				&& BvhMath.AsUint( a.AabbMin.z ) == BvhMath.AsUint( b.AabbMin.z )
				&& a.LeftFirst == b.LeftFirst
				&& BvhMath.AsUint( a.AabbMax.x ) == BvhMath.AsUint( b.AabbMax.x )
				&& BvhMath.AsUint( a.AabbMax.y ) == BvhMath.AsUint( b.AabbMax.y )
				&& BvhMath.AsUint( a.AabbMax.z ) == BvhMath.AsUint( b.AabbMax.z )
				&& a.TriCount == b.TriCount;
		}

		/// <summary>Bit-exact node-pool comparison, replacing the Unity test's UnsafeUtility.MemCmp on Nodes.</summary>
		static void AssertNodesEqual( Bvh a, Bvh b )
		{
			for ( uint i = 0; i < a.UsedNodes; i++ )
			{
				Assert.IsTrue( NodeBitsEqual( a.Nodes[ i ], b.Nodes[ i ] ), $"nodes[{i}]" );
			}
		}

		/// <summary>Bit-exact index-array comparison, replacing the Unity test's UnsafeUtility.MemCmp on PrimIdx.</summary>
		static void AssertPrimIdxEqual( Bvh a, Bvh b )
		{
			for ( uint i = 0; i < a.TriCount; i++ )
			{
				Assert.AreEqual( a.PrimIdx[ i ], b.PrimIdx[ i ], $"primIdx[{i}]" );
			}
		}

		/// <summary>Every fragment referenced by a leaf must fit inside the bounds of that leaf.</summary>
		private static void AssertLeafContainment( Bvh bvh )
		{
			Span<uint> stack = stackalloc uint[ 64 ];
			int stackPtr = 0;
			uint nodeIdx = 0, leaves = 0, prims = 0;
			while ( true )
			{
				BvhNode node = bvh.Nodes[ nodeIdx ];
				if ( node.IsLeaf )
				{
					leaves++;
					for ( uint i = 0; i < node.TriCount; i++ )
					{
						uint prim = bvh.PrimIdx[ node.LeftFirst + i ];
						Sphere s = SphereGeometry.SphereAt( prim );
						BvhVec3 bmin = s.Center - new BvhVec3( s.Radius ), bmax = s.Center + new BvhVec3( s.Radius );
						Assert.IsTrue(
							bmin.x >= node.AabbMin.x && bmin.y >= node.AabbMin.y && bmin.z >= node.AabbMin.z &&
							bmax.x <= node.AabbMax.x && bmax.y <= node.AabbMax.y && bmax.z <= node.AabbMax.z,
							$"fragment {prim} is outside the bounds of its leaf" );
						prims++;
					}
					if ( stackPtr == 0 )
					{
						break;
					}
					nodeIdx = stack[ --stackPtr ];
				}
				else
				{
					nodeIdx = node.LeftFirst;
					stack[ stackPtr++ ] = node.LeftFirst + 1;
				}
			}
			Assert.AreEqual( SphereCount, prims, "every primitive should end up in exactly one leaf" );
			TestContext.WriteLine( $"custom bvh: {bvh.UsedNodes} nodes, {leaves} leaves" );
		}

		[Test]
		public void BuildAabbs_AndCallbackBuild_ProduceTheSameTree()
		{
			BvhVec4[] aabbs = BuildAabbArray();
			Bvh fromArray = new Bvh();
			Bvh fromCallback = new Bvh();
			fromArray.BuildAabbs( aabbs, SphereCount );
			fromCallback.Build( SphereGeometry.GetAabb, SphereCount );

			Assert.IsNull( fromArray.VertIdx, "a BVH over AABBs has no vertex indices" );
			Assert.IsNull( fromArray.Verts, "a BVH over AABBs has no vertices" );
			Assert.AreEqual( SphereCount, fromArray.TriCount );

			Assert.AreEqual( fromArray.UsedNodes, fromCallback.UsedNodes, "node count" );
			AssertNodesEqual( fromArray, fromCallback );
			AssertPrimIdxEqual( fromArray, fromCallback );

			AssertLeafContainment( fromArray );
		}

		[Test]
		public void CustomCallbacks_MatchBruteForce()
		{
			BvhVec4[] aabbs = BuildAabbArray();
			BvhVec3[] origins = new BvhVec3[ RayCount ];
			BvhVec3[] directions = new BvhVec3[ RayCount ];
			Intersection[] hits = new Intersection[ RayCount ];
			int[] occluded = new int[ RayCount ];
			Intersection[] bruteHits = new Intersection[ RayCount ];
			int[] bruteOccluded = new int[ RayCount ];
			Bvh bvh = new Bvh();
			bvh.BuildAabbs( aabbs, SphereCount );
			bvh.CustomIntersect = SphereGeometry.Intersect;
			bvh.CustomIsOccluded = SphereGeometry.IsOccluded;
			MakeRays( origins, directions );

			TraceSpheres( bvh, origins, directions, hits, occluded );
			TraceBruteForce( bvh.CustomIntersect, bvh.CustomIsOccluded, origins, directions, bruteHits, bruteOccluded );

			int mismatches = 0, ties = 0, occlusionMismatches = 0, hitCount = 0;
			for ( int i = 0; i < RayCount; i++ )
			{
				Intersection hit = hits[ i ], brute = bruteHits[ i ];
				if ( hit.T != brute.T )
				{
					mismatches++;
				}
				else if ( hit.T < BvhConstants.Far )
				{
					hitCount++;
					if ( hit.Prim != brute.Prim )
					{
						ties++;
					}
				}
				if ( occluded[ i ] != bruteOccluded[ i ] )
				{
					occlusionMismatches++;
				}
			}
			TestContext.WriteLine( $"custom spheres: {hitCount}/{RayCount} hits, mismatches {mismatches}, same-distance ties {ties}, occlusion mismatches {occlusionMismatches}" );
			Assert.AreEqual( 0, mismatches, "intersect mismatches" );
			Assert.AreEqual( 0, occlusionMismatches, "occlusion mismatches" );
			Assert.Greater( hitCount, RayCount / 10, "the rays should mostly hit the sphere cloud" );
		}
	}
}
