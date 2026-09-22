using System;
using NUnit.Framework;

namespace MicroBVH.Tests
{
	/// <summary>Self-contained BVH tests that need no external data files.</summary>
	public class BvhBasicTests
	{
		const int GridSize = 12;

		/// <summary>Deterministic LCG, same recurrence as tinybvh's reference generators.</summary>
		struct Lcg
		{
			uint state;

			public Lcg( uint seed )
			{
				state = seed;
			}

			public float NextFloat()
			{
				state = ( state * 1664525u ) + 1013904223u;
				return ( state >> 8 ) * ( 1f / 16777216f );
			}
		}

		/// <summary>Port of Möller-Trumbore, used as a brute-force cross-check against BVH traversal.</summary>
		static bool RayTriangle( BvhVec3 O, BvhVec3 D, BvhVec3 v0, BvhVec3 v1, BvhVec3 v2, out float t, out float u, out float v )
		{
			t = 0f;
			u = 0f;
			v = 0f;

			BvhVec3 e1 = v1 - v0;
			BvhVec3 e2 = v2 - v0;
			BvhVec3 h = BvhMath.Cross( D, e2 );
			float a = BvhMath.Dot( e1, h );
			if ( MathF.Abs( a ) < 1e-12f )
			{
				return false;
			}

			float f = 1f / a;
			BvhVec3 s = O - v0;
			u = f * BvhMath.Dot( s, h );
			if ( u < 0f || u > 1f )
			{
				return false;
			}

			BvhVec3 q = BvhMath.Cross( s, e1 );
			v = f * BvhMath.Dot( D, q );
			if ( v < 0f || u + v > 1f )
			{
				return false;
			}

			t = f * BvhMath.Dot( e2, q );
			return t > 1e-6f;
		}

		/// <summary>Port of TestIndexedGeometry's grid mesh (tiny_bvh_double_test.cpp), in single precision.</summary>
		static void BuildGrid( out BvhVec3[] gridVerts, out uint[] indices, out BvhVec4[] soup, out int triCount )
		{
			int quads = ( GridSize - 1 ) * ( GridSize - 1 );
			triCount = quads * 2;

			gridVerts = new BvhVec3[ GridSize * GridSize ];
			for ( int j = 0; j < GridSize; j++ )
			{
				for ( int i = 0; i < GridSize; i++ )
				{
					float z = 0.05f * ( ( i * 7 + j * 13 ) % 5 );
					gridVerts[ ( j * GridSize ) + i ] = new BvhVec3( i, j, z );
				}
			}

			indices = new uint[ triCount * 3 ];
			soup = new BvhVec4[ triCount * 3 ];
			int t = 0;
			for ( int j = 0; j < GridSize - 1; j++ )
			{
				for ( int i = 0; i < GridSize - 1; i++ )
				{
					uint a = ( uint )( ( j * GridSize ) + i );
					uint b = ( uint )( ( j * GridSize ) + i + 1 );
					uint c = ( uint )( ( ( j + 1 ) * GridSize ) + i );
					uint d = ( uint )( ( ( j + 1 ) * GridSize ) + i + 1 );
					uint[][] tris = new uint[][] { new uint[] { a, b, c }, new uint[] { b, d, c } };
					for ( int k = 0; k < 2; k++, t++ )
					{
						for ( int v = 0; v < 3; v++ )
						{
							indices[ ( t * 3 ) + v ] = tris[ k ][ v ];
							soup[ ( t * 3 ) + v ] = new BvhVec4( gridVerts[ tris[ k ][ v ] ], 0f );
						}
					}
				}
			}
		}

		[Test]
		public void RandomTriangles_BuildAndIntersect()
		{
			const uint triCount = 8192;
			BvhVec4[] verts = new BvhVec4[ ( int )triCount * 3 ];
			Bvh bvh = new Bvh();
			Lcg rng = new Lcg( 0x12345678u );
			for ( int i = 0; i < triCount; i++ )
			{
				BvhVec3 basePos = new BvhVec3( rng.NextFloat(), rng.NextFloat(), rng.NextFloat() );
				for ( int v = 0; v < 3; v++ )
				{
					BvhVec3 jitter = new BvhVec3( rng.NextFloat(), rng.NextFloat(), rng.NextFloat() ) * 0.1f;
					verts[ ( i * 3 ) + v ] = new BvhVec4( basePos + jitter, 0f );
				}
			}

			bvh.Build( verts, triCount );

			Ray ray = new Ray( new BvhVec3( 0.5f, 0.5f, -1f ), new BvhVec3( 0.1f, 0f, 2f ) );
			bvh.Intersect( ref ray );

			Assert.Less( ray.Hit.T, BvhConstants.Far );

			float bestT = BvhConstants.Far;
			uint bestPrim = uint.MaxValue;
			for ( uint i = 0; i < triCount; i++ )
			{
				BvhVec3 v0 = verts[ ( int )( i * 3 ) ];
				BvhVec3 v1 = verts[ ( int )( ( i * 3 ) + 1 ) ];
				BvhVec3 v2 = verts[ ( int )( ( i * 3 ) + 2 ) ];
				if ( RayTriangle( ray.O, ray.D, v0, v1, v2, out float t, out _, out _ ) && t < bestT )
				{
					bestT = t;
					bestPrim = i;
				}
			}

			Assert.AreEqual( bestPrim, ray.Hit.Prim );
			Assert.That( ray.Hit.T, Is.EqualTo( bestT ).Within( 1e-5f * bestT ) );

			Assert.Greater( bvh.NodeCount(), 1 );
			Assert.LessOrEqual( bvh.UsedNodes, 2 * triCount );

			const float eps = 1e-4f;
			for ( uint n = 0; n < bvh.UsedNodes; n++ )
			{
				BvhNode node = bvh.Nodes[ n ];
				if ( node.TriCount == 0 )
				{
					continue;
				}
				for ( uint k = 0; k < node.TriCount; k++ )
				{
					uint prim = bvh.PrimIdx[ node.LeftFirst + k ];
					for ( int v = 0; v < 3; v++ )
					{
						BvhVec3 p = verts[ ( int )( ( prim * 3 ) + v ) ];
						Assert.GreaterOrEqual( p.x, node.AabbMin.x - eps );
						Assert.GreaterOrEqual( p.y, node.AabbMin.y - eps );
						Assert.GreaterOrEqual( p.z, node.AabbMin.z - eps );
						Assert.LessOrEqual( p.x, node.AabbMax.x + eps );
						Assert.LessOrEqual( p.y, node.AabbMax.y + eps );
						Assert.LessOrEqual( p.z, node.AabbMax.z + eps );
					}
				}
			}
		}

		[Test]
		public void IndexedVsSoup_GridMesh()
		{
			BuildGrid( out BvhVec3[] gridVerts, out uint[] indices, out BvhVec4[] soup, out int triCount );

			BvhVec4[] vertsAsFloat4 = new BvhVec4[ gridVerts.Length ];
			for ( int i = 0; i < gridVerts.Length; i++ )
			{
				vertsAsFloat4[ i ] = new BvhVec4( gridVerts[ i ], 0f );
			}

			Bvh indexedBvh = new Bvh();
			Bvh plainBvh = new Bvh();
			indexedBvh.Build( vertsAsFloat4, indices, ( uint )triCount );
			plainBvh.Build( soup, ( uint )triCount );

			int hits = 0;
			for ( int sy = 0; sy < 40; sy++ )
			{
				for ( int sx = 0; sx < 40; sx++ )
				{
					float px = -1f + ( ( float )sx / 39f * ( GridSize + 1 ) );
					float py = -1f + ( ( float )sy / 39f * ( GridSize + 1 ) );
					BvhVec3 O = new BvhVec3( px, py, -5f );
					BvhVec3 D = new BvhVec3( 0f, 0f, 1f );

					Ray ri = new Ray( O, D );
					Ray rp = new Ray( O, D );
					indexedBvh.Intersect( ref ri );
					plainBvh.Intersect( ref rp );

					bool hitI = ri.Hit.T < BvhConstants.Far;
					bool hitP = rp.Hit.T < BvhConstants.Far;
					Assert.AreEqual( hitP, hitI );
					if ( hitI )
					{
						hits++;
						Assert.AreEqual( rp.Hit.Prim, ri.Hit.Prim );
						Assert.That( ri.Hit.T, Is.EqualTo( rp.Hit.T ).Within( 1e-6f ) );
						Assert.That( ri.Hit.U, Is.EqualTo( rp.Hit.U ).Within( 1e-6f ) );
						Assert.That( ri.Hit.V, Is.EqualTo( rp.Hit.V ).Within( 1e-6f ) );
					}

					Ray si = new Ray( O, D );
					Ray sp = new Ray( O, D );
					bool oi = indexedBvh.IsOccluded( si );
					bool op = plainBvh.IsOccluded( sp );
					Assert.AreEqual( op, oi );
					Assert.AreEqual( hitI, oi );
				}
			}

			Assert.Greater( hits, 0 );
		}

		[Test]
		public void Refit_MovesBounds()
		{
			BuildGrid( out _, out _, out BvhVec4[] soup, out int triCount );

			Bvh bvh = new Bvh();
			bvh.Build( soup, ( uint )triCount );

			BvhVec3 originalMin = bvh.AabbMin;
			BvhVec3 originalMax = bvh.AabbMax;

			BvhVec3 O = new BvhVec3( 5f, 5f, -5f );
			BvhVec3 D = new BvhVec3( 0f, 0f, 1f );
			Ray rayBefore = new Ray( O, D );
			bvh.Intersect( ref rayBefore );
			Assert.Less( rayBefore.Hit.T, BvhConstants.Far );

			for ( int i = 0; i < soup.Length; i++ )
			{
				BvhVec4 v = soup[ i ];
				v.y += 20f;
				soup[ i ] = v;
			}
			bvh.Refit();

			Assert.That( bvh.AabbMin.y, Is.EqualTo( originalMin.y + 20f ).Within( 1e-4f ) );
			Assert.That( bvh.AabbMax.y, Is.EqualTo( originalMax.y + 20f ).Within( 1e-4f ) );
			Assert.That( bvh.AabbMin.x, Is.EqualTo( originalMin.x ).Within( 1e-4f ) );
			Assert.That( bvh.AabbMax.x, Is.EqualTo( originalMax.x ).Within( 1e-4f ) );

			Ray rayAfterSame = new Ray( O, D );
			bvh.Intersect( ref rayAfterSame );
			Assert.AreEqual( BvhConstants.Far, rayAfterSame.Hit.T );

			Ray rayShifted = new Ray( new BvhVec3( 5f, 25f, -5f ), D );
			bvh.Intersect( ref rayShifted );
			Assert.Less( rayShifted.Hit.T, BvhConstants.Far );
		}

		[Test]
		public void Tlas_TwoInstances()
		{
			BuildGrid( out _, out _, out BvhVec4[] soup, out int triCount );

			Bvh blas = new Bvh();
			blas.Build( soup, ( uint )triCount );
			Bvh[] blasses = new Bvh[] { blas };

			BlasInstance[] instArray = new BlasInstance[ 2 ];
			BlasInstance inst0 = BlasInstance.Create( 0 );
			BlasInstance inst1 = BlasInstance.Create( 0 );
			inst1.Transform[ 3 ] = 10f; // row-major cell 3 = Row0.w: translate +10 along x.
			inst1.Mask = 0x2;
			instArray[ 0 ] = inst0;
			instArray[ 1 ] = inst1;

			Bvh tlas = new Bvh();
			tlas.BuildTlas( instArray, 2, blasses, 1 );

			BvhVec3 O = new BvhVec3( 5f, 5f, -5f );
			BvhVec3 D = new BvhVec3( 0f, 0f, 1f );

			Ray localRay = new Ray( O, D );
			blas.Intersect( ref localRay );
			Assert.Less( localRay.Hit.T, BvhConstants.Far );

			Ray ray0 = new Ray( O, D );
			tlas.Intersect( ref ray0 );
			Assert.Less( ray0.Hit.T, BvhConstants.Far );
			Assert.AreEqual( 0u, ray0.Hit.Inst );
			Assert.AreEqual( localRay.Hit.Prim, ray0.Hit.Prim );

			BvhVec3 O1 = new BvhVec3( 15f, 5f, -5f );
			Ray ray1 = new Ray( O1, D );
			tlas.Intersect( ref ray1 );
			Assert.Less( ray1.Hit.T, BvhConstants.Far );
			Assert.AreEqual( 1u, ray1.Hit.Inst );
			Assert.AreEqual( localRay.Hit.Prim, ray1.Hit.Prim );

			// Instance 1's mask is 0x2; a ray restricted to 0x1 must miss it entirely.
			Ray maskedRay = new Ray( O1, D, BvhConstants.Far, 0x1 );
			tlas.Intersect( ref maskedRay );
			Assert.AreEqual( BvhConstants.Far, maskedRay.Hit.T );
		}

		[Test]
		public void IsOccluded_RespectsMaxDistance()
		{
			BuildGrid( out _, out _, out BvhVec4[] soup, out int triCount );

			Bvh bvh = new Bvh();
			bvh.Build( soup, ( uint )triCount );

			BvhVec3 O = new BvhVec3( 5f, 5f, -5f );
			BvhVec3 D = new BvhVec3( 0f, 0f, 1f );
			Ray probe = new Ray( O, D );
			bvh.Intersect( ref probe );
			Assert.Less( probe.Hit.T, BvhConstants.Far );
			float t = probe.Hit.T;

			Ray shortRay = new Ray( O, D, 0.5f * t );
			Assert.IsFalse( bvh.IsOccluded( shortRay ) );

			Ray fullRay = new Ray( O, D, BvhConstants.Far );
			Assert.IsTrue( bvh.IsOccluded( fullRay ) );
		}
	}
}
