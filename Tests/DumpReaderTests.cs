using System.IO;
using NUnit.Framework;

namespace MicroBVH.Tests
{
	/// <summary>
	/// Sanity checks for the reference-dump reader (RefDumpFile): the file loads without throwing
	/// (which also means its magic was accepted and, since Load checks it is left at end of
	/// stream, that the reader consumed the whole file), and key counts are non-zero and mutually
	/// consistent. This does not compare against the C++ reference values themselves; that is done
	/// by the tests that exercise the corresponding library code.
	/// </summary>
	public class DumpReaderTests
	{
		[TestCase( "suzanne" )]
		[TestCase( "bunny" )]
		[TestCase( "cryteksponza" )]
		public void RefDump_IsWellFormed( string scene )
		{
			string path = BvhSceneFile.TestDataPath( scene + ".ref" );
			if ( !File.Exists( path ) )
			{
				Assert.Ignore( $"missing {path}; see the README for how to generate the reference dumps" );
			}
			RefDumpFile file = RefDumpFile.Load( path );

			Assert.Greater( file.TriCount, 0u );

			Assert.Greater( file.UsedNodes, 0u );
			Assert.AreEqual( file.UsedNodes, ( uint )file.Nodes.Length );
			Assert.Greater( file.PrimIdx.Length, 0 );
			Assert.AreEqual( 65536, file.Rays.Length );
			Assert.AreEqual( 65536, file.RefitHits.Length );

			Assert.AreEqual( 3, file.Instances.Length );
			Assert.Greater( file.TlasUsedNodes, 0u );
			Assert.AreEqual( file.TlasUsedNodes, ( uint )file.TlasNodes.Length );
			Assert.AreEqual( 65536, file.TlasRays.Length );

			Assert.Greater( file.WeldedVertices.Length, 0 );
			Assert.AreEqual( file.TriCount * 3, ( uint )file.Indices.Length );
			Assert.Greater( file.IndexedUsedNodes, 0u );
			Assert.AreEqual( file.IndexedUsedNodes, ( uint )file.IndexedNodes.Length );
			Assert.AreEqual( 65536, file.IndexedRays.Length );
		}
	}
}
