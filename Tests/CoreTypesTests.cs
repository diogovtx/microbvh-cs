using System.IO;
using NUnit.Framework;

namespace MicroBVH.Tests
{
	/// <summary>Checks of the vector and matrix helpers that every other section builds on.</summary>
	public class CoreTypesTests
	{
		[Test]
		public void MinMax_FollowTheCppTernaries()
		{
			// tinybvh_min( a, b ) is a < b ? a : b, so a NaN in either argument yields b, and
			// min( -0, +0 ) yields +0. Math.Min would give NaN and -0.
			Assert.AreEqual( 1f, BvhMath.Min( float.NaN, 1f ) );
			Assert.IsTrue( float.IsNaN( BvhMath.Min( 1f, float.NaN ) ) );
			Assert.AreEqual( 0u, BvhMath.AsUint( BvhMath.Min( -0f, 0f ) ) );
			Assert.AreEqual( 0x80000000u, BvhMath.AsUint( BvhMath.Max( 0f, -0f ) ) );
		}

		[Test]
		public void Rcp_CapsNonFiniteReciprocals()
		{
			Assert.AreEqual( BvhConstants.RcpMax, BvhMath.Rcp( 0f ) );
			Assert.AreEqual( -BvhConstants.RcpMax, BvhMath.Rcp( -0f ) );
			Assert.AreEqual( 0.5f, BvhMath.Rcp( 2f ) );
		}

		[Test]
		public void Mat4_CellOrderAndInverse()
		{
			BlasInstance inst = BlasInstance.Create( 0 );
			inst.Transform[ 3 ] = 2f; // translation x
			inst.Transform[ 7 ] = 3f; // translation y
			inst.Transform[ 0 ] = 4f; // scale x
			Assert.AreEqual( 2f, inst.Transform.Row0.w );
			Assert.AreEqual( 3f, inst.Transform.Row1.w );
			inst.InvertTransform();
			BvhVec3 p = inst.Transform.TransformPoint( new BvhVec3( 1f, 1f, 1f ) );
			Assert.AreEqual( 6f, p.x );
			Assert.AreEqual( 4f, p.y );
			BvhVec3 q = inst.InvTransform.TransformPoint( p );
			Assert.AreEqual( 1f, q.x, 1e-6f );
			Assert.AreEqual( 1f, q.y, 1e-6f );
			Assert.AreEqual( 1f, q.z, 1e-6f );
		}

		[Test]
		public void SceneFile_LoadsFromTheDataDirectory()
		{
			string binPath = BvhSceneFile.TestDataPath( "suzanne.bin" );
			if ( !File.Exists( binPath ) )
			{
				Assert.Ignore( $"missing {binPath}; run TestData/fetch.ps1 for the scenes" );
			}
			BvhVec4[] verts = BvhSceneFile.Load( binPath, out uint triCount );
			Assert.Greater( triCount, 0u );
			Assert.AreEqual( triCount * 3, ( uint )verts.Length );
		}

		[Test]
		public void Ray_NormalizesAndSetsUpReciprocal()
		{
			Ray ray = new Ray( new BvhVec3( 0f ), new BvhVec3( 0f, 0f, 2f ) );
			Assert.AreEqual( 1f, ray.D.z );
			Assert.AreEqual( 1f, ray.RD.z );
			Assert.AreEqual( BvhConstants.RcpMax, ray.RD.x );
			Assert.AreEqual( BvhConstants.Far, ray.Hit.T );
			Assert.AreEqual( BvhConstants.RayMaskIntersectAll, ray.Mask );
		}
	}
}
