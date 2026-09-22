using System;
using System.IO;
using System.Runtime.CompilerServices;
using NUnit.Framework;

namespace MicroBVH.Tests
{
	/// <summary>
	/// Loader for tinybvh's ".bin" test scenes (int32 triCount followed by triCount * 3 float4
	/// vertices) and the location of the reference data. The data directory is MICROBVH_TESTDATA
	/// when set; otherwise the TestData folder at the root of this repository, where the README's
	/// instructions put the scenes and the reference dumps.
	/// </summary>
	public static class BvhSceneFile
	{
		private static string dataDir;

		/// <summary>The directory holding the scenes and reference dumps.</summary>
		public static string DataDirectory
		{
			get
			{
				if ( dataDir == null )
				{
					dataDir = ResolveDataDirectory();
				}
				return dataDir;
			}
		}

		/// <summary>Resolves a file name to a path in <see cref="DataDirectory"/>.</summary>
		public static string TestDataPath( string fileName )
		{
			return Path.Combine( DataDirectory, fileName );
		}

		/// <summary>Reads a scene file into a new array of triCount * 3 vertices.</summary>
		public static BvhVec4[] Load( string path, out uint triCount )
		{
			byte[] bytes = File.ReadAllBytes( path );
			int count = BitConverter.ToInt32( bytes, 0 );
			triCount = ( uint )count;
			BvhVec4[] verts = new BvhVec4[ count * 3 ];
			for ( int i = 0; i < verts.Length; i++ )
			{
				int offset = 4 + ( i * 16 );
				verts[ i ] = new BvhVec4(
					BitConverter.ToSingle( bytes, offset ),
					BitConverter.ToSingle( bytes, offset + 4 ),
					BitConverter.ToSingle( bytes, offset + 8 ),
					BitConverter.ToSingle( bytes, offset + 12 ) );
			}
			return verts;
		}

		private static string ResolveDataDirectory( [CallerFilePath] string sourcePath = "" )
		{
			string fromEnvironment = Environment.GetEnvironmentVariable( "MICROBVH_TESTDATA" );
			if ( !string.IsNullOrEmpty( fromEnvironment ) )
			{
				return fromEnvironment;
			}
			// find the repository root, recognized by MicroBVH.csproj: above this source file when the
			// tests run from a local build, otherwise above the test binaries.
			DirectoryInfo dir = FindRepositoryRoot( Path.GetDirectoryName( sourcePath ) ) ?? FindRepositoryRoot( TestContext.CurrentContext.TestDirectory );
			if ( dir == null )
			{
				return Path.Combine( TestContext.CurrentContext.TestDirectory, "TestData" );
			}
			return Path.Combine( dir.FullName, "TestData" );
		}

		private static DirectoryInfo FindRepositoryRoot( string start )
		{
			if ( string.IsNullOrEmpty( start ) || !Directory.Exists( start ) )
			{
				return null;
			}
			DirectoryInfo dir = new DirectoryInfo( start );
			while ( dir != null && !File.Exists( Path.Combine( dir.FullName, "MicroBVH.csproj" ) ) )
			{
				dir = dir.Parent;
			}
			return dir;
		}
	}
}
