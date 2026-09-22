using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace MicroBVH.Benchmarks
{
	/// <summary>
	/// Stopwatch benchmark for MicroBVH.cs: build and traversal throughput over tinybvh's suzanne,
	/// bunny and cryteksponza test scenes. Scene files are resolved like the tests: the first
	/// command-line argument, then MICROBVH_TESTDATA, then the TestData folder at the repository
	/// root. A second argument overrides the ray count, a third the number of timed runs.
	/// </summary>
	internal static class Program
	{
		/// <summary>Rays per scene.</summary>
		const int DefaultRayCount = 1 << 20;
		/// <summary>Timed runs per measurement, taken after one untimed warm-up run.</summary>
		const int DefaultRunCount = 5;
		/// <summary>Rays handed to one Parallel.ForEach delegate invocation.</summary>
		const int RayChunkSize = 4096;
		/// <summary>Seed for the deterministic ray generator.</summary>
		const uint RaySeed = 0x1234ABCDu;

		private static readonly string[] sceneNames = { "suzanne", "bunny", "cryteksponza" };

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

		static void Main( string[] args )
		{
			string dataDir = ResolveDataDirectory( args );
			int rayCount = DefaultRayCount;
			if ( args.Length > 1 && int.TryParse( args[ 1 ], NumberStyles.Integer, CultureInfo.InvariantCulture, out int rayArg ) && rayArg > 0 )
			{
				rayCount = rayArg;
			}
			int runCount = DefaultRunCount;
			if ( args.Length > 2 && int.TryParse( args[ 2 ], NumberStyles.Integer, CultureInfo.InvariantCulture, out int runArg ) && runArg > 0 )
			{
				runCount = runArg;
			}

			Console.WriteLine( FormatMachineLine() );
			Console.WriteLine();

			List<string> rows = new List<string>();
			foreach ( string sceneName in sceneNames )
			{
				string row = RunScene( sceneName, dataDir, rayCount, runCount );
				if ( row != null )
				{
					rows.Add( row );
				}
			}

			int processorCount = Environment.ProcessorCount;
			Console.WriteLine( $"| Scene | Triangles | Nodes | Build (ms) | Threaded build (ms) | Intersect, 1 thread (Mrays/s) | Intersect, {processorCount} threads (Mrays/s) | IsOccluded, 1 thread (Mrays/s) | IsOccluded, {processorCount} threads (Mrays/s) |" );
			Console.WriteLine( "| --- | --- | --- | --- | --- | --- | --- | --- | --- |" );
			foreach ( string row in rows )
			{
				Console.WriteLine( row );
			}
		}

		/// <summary>First command-line argument, else MICROBVH_TESTDATA, else TestData at the repository root.</summary>
		static string ResolveDataDirectory( string[] args, [CallerFilePath] string sourcePath = "" )
		{
			if ( args.Length > 0 && !string.IsNullOrEmpty( args[ 0 ] ) )
			{
				return args[ 0 ];
			}
			string fromEnvironment = Environment.GetEnvironmentVariable( "MICROBVH_TESTDATA" );
			if ( !string.IsNullOrEmpty( fromEnvironment ) )
			{
				return fromEnvironment;
			}
			// the repository root, recognized by MicroBVH.csproj: above this source file when run from a
			// local build, otherwise above the binaries.
			DirectoryInfo dir = FindRepositoryRoot( Path.GetDirectoryName( sourcePath ) ) ?? FindRepositoryRoot( AppContext.BaseDirectory );
			string root = dir != null ? dir.FullName : AppContext.BaseDirectory;
			return Path.Combine( root, "TestData" );
		}

		static DirectoryInfo FindRepositoryRoot( string start )
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

		/// <summary>Runs every measurement for one scene and returns its Markdown row, or null when the scene file is missing.</summary>
		static string RunScene( string sceneName, string dataDir, int rayCount, int runCount )
		{
			string path = Path.Combine( dataDir, sceneName + ".bin" );
			if ( !File.Exists( path ) )
			{
				Console.Error.WriteLine( $"{sceneName}: {path} not found; run TestData/fetch.ps1 to fetch the test scenes. Skipping." );
				return null;
			}

			Console.Error.WriteLine( $"{sceneName}: loading" );
			BvhVec4[] verts = LoadScene( path, out uint triCount );

			Console.Error.WriteLine( $"{sceneName}: build, serial" );
			Bvh bvh = new Bvh();
			double buildMs = Median( TimeBuild( bvh, verts, triCount, false, runCount ) );
			int nodeCount = bvh.NodeCount();

			string threadedBuildCell;
			if ( triCount >= BvhConstants.MtBuildThreshold )
			{
				Console.Error.WriteLine( $"{sceneName}: build, threaded" );
				Bvh threadedBvh = new Bvh();
				double threadedBuildMs = Median( TimeBuild( threadedBvh, verts, triCount, true, runCount ) );
				threadedBuildCell = FormatMs( threadedBuildMs );
			}
			else
			{
				threadedBuildCell = "-";
			}

			Ray[] pristineRays = GenerateRays( bvh.AabbMin, bvh.AabbMax, rayCount, RaySeed );

			Console.Error.WriteLine( $"{sceneName}: intersect, 1 thread" );
			double intersectMs1 = Median( TimeIntersect( bvh, pristineRays, runCount, false ) );
			Console.Error.WriteLine( $"{sceneName}: intersect, {Environment.ProcessorCount} threads" );
			double intersectMsN = Median( TimeIntersect( bvh, pristineRays, runCount, true ) );
			Console.Error.WriteLine( $"{sceneName}: isOccluded, 1 thread" );
			double occludedMs1 = Median( TimeIsOccluded( bvh, pristineRays, runCount, false ) );
			Console.Error.WriteLine( $"{sceneName}: isOccluded, {Environment.ProcessorCount} threads" );
			double occludedMsN = Median( TimeIsOccluded( bvh, pristineRays, runCount, true ) );

			double hitFraction = HitFraction( bvh, pristineRays );
			string hitPercent = ( hitFraction * 100.0 ).ToString( "F1", CultureInfo.InvariantCulture );
			Console.Error.WriteLine( $"{sceneName}: intersect hit rate {hitPercent}% (sanity check)" );

			return $"| {sceneName} | {triCount} | {nodeCount} | {FormatMs( buildMs )} | {threadedBuildCell} | {FormatMrays( MraysPerSecond( rayCount, intersectMs1 ) )} | {FormatMrays( MraysPerSecond( rayCount, intersectMsN ) )} | {FormatMrays( MraysPerSecond( rayCount, occludedMs1 ) )} | {FormatMrays( MraysPerSecond( rayCount, occludedMsN ) )} |";
		}

		/// <summary>Reads a tinybvh scene file: int32 triCount, then triCount * 3 float4 vertices.</summary>
		static BvhVec4[] LoadScene( string path, out uint triCount )
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

		/// <summary>
		/// Builds bvh runCount + 1 times, one untimed warm-up run first, and returns the elapsed
		/// milliseconds of the timed runs. Rebuilding the same instance lets it reuse its node pool
		/// after the warm-up, so the timed runs measure construction rather than allocation.
		/// </summary>
		static double[] TimeBuild( Bvh bvh, BvhVec4[] verts, uint triCount, bool threaded, int runCount )
		{
			bvh.UseThreadedBuild = threaded;
			double[] times = new double[ runCount ];
			Stopwatch sw = new Stopwatch();
			for ( int i = -1; i < runCount; i++ )
			{
				sw.Restart();
				bvh.Build( verts, triCount );
				sw.Stop();
				if ( i >= 0 )
				{
					times[ i ] = sw.Elapsed.TotalMilliseconds;
				}
			}
			return times;
		}

		/// <summary>
		/// Generates count deterministic rays around a scene: the origin sits outside the bounds
		/// along a random direction from the center, the target is a random point inside the bounds.
		/// </summary>
		static Ray[] GenerateRays( BvhVec3 aabbMin, BvhVec3 aabbMax, int count, uint seed )
		{
			BvhVec3 center = ( aabbMin + aabbMax ) * 0.5f;
			BvhVec3 extent = aabbMax - aabbMin;
			float originDistance = BvhMath.Length( extent ) * 0.5f * 1.2f;

			Lcg rng = new Lcg( seed );
			Ray[] rays = new Ray[ count ];
			for ( int i = 0; i < count; i++ )
			{
				BvhVec3 origin = center + ( RandomUnitVector( ref rng ) * originDistance );
				BvhVec3 target = aabbMin + ( extent * new BvhVec3( rng.NextFloat(), rng.NextFloat(), rng.NextFloat() ) );
				rays[ i ] = new Ray( origin, target - origin );
			}
			return rays;
		}

		/// <summary>Uniformly distributed point on the unit sphere, from two LCG draws.</summary>
		static BvhVec3 RandomUnitVector( ref Lcg rng )
		{
			float z = ( 2f * rng.NextFloat() ) - 1f;
			float r = MathF.Sqrt( MathF.Max( 0f, 1f - ( z * z ) ) );
			float phi = 2f * MathF.PI * rng.NextFloat();
			return new BvhVec3( r * MathF.Cos( phi ), r * MathF.Sin( phi ), z );
		}

		/// <summary>
		/// Times runCount + 1 Intersect passes, one untimed warm-up first, over a fresh copy of
		/// pristineRays each time, since Intersect writes the hit back into ray.Hit.
		/// </summary>
		static double[] TimeIntersect( Bvh bvh, Ray[] pristineRays, int runCount, bool threaded )
		{
			int count = pristineRays.Length;
			Ray[] working = new Ray[ count ];
			double[] times = new double[ runCount ];
			Stopwatch sw = new Stopwatch();
			for ( int i = -1; i < runCount; i++ )
			{
				Array.Copy( pristineRays, working, count );
				sw.Restart();
				if ( threaded )
				{
					Parallel.ForEach( Partitioner.Create( 0, count, RayChunkSize ), range =>
					{
						for ( int r = range.Item1; r < range.Item2; r++ )
						{
							bvh.Intersect( ref working[ r ] );
						}
					} );
				}
				else
				{
					for ( int r = 0; r < count; r++ )
					{
						bvh.Intersect( ref working[ r ] );
					}
				}
				sw.Stop();
				if ( i >= 0 )
				{
					times[ i ] = sw.Elapsed.TotalMilliseconds;
				}
			}
			return times;
		}

		/// <summary>
		/// As TimeIntersect, for IsOccluded. IsOccluded does not write to the ray, but it reads
		/// ray.Hit.T as the search distance, so it still needs a fresh, un-shortened copy every run.
		/// </summary>
		static double[] TimeIsOccluded( Bvh bvh, Ray[] pristineRays, int runCount, bool threaded )
		{
			int count = pristineRays.Length;
			Ray[] working = new Ray[ count ];
			double[] times = new double[ runCount ];
			Stopwatch sw = new Stopwatch();
			for ( int i = -1; i < runCount; i++ )
			{
				Array.Copy( pristineRays, working, count );
				sw.Restart();
				if ( threaded )
				{
					Parallel.ForEach( Partitioner.Create( 0, count, RayChunkSize ), range =>
					{
						for ( int r = range.Item1; r < range.Item2; r++ )
						{
							bvh.IsOccluded( in working[ r ] );
						}
					} );
				}
				else
				{
					for ( int r = 0; r < count; r++ )
					{
						bvh.IsOccluded( in working[ r ] );
					}
				}
				sw.Stop();
				if ( i >= 0 )
				{
					times[ i ] = sw.Elapsed.TotalMilliseconds;
				}
			}
			return times;
		}

		/// <summary>Fraction of pristineRays that Intersect reports a hit for; a sanity check, not a timed measurement.</summary>
		static double HitFraction( Bvh bvh, Ray[] pristineRays )
		{
			int count = pristineRays.Length;
			Ray[] working = new Ray[ count ];
			Array.Copy( pristineRays, working, count );
			int hits = 0;
			for ( int r = 0; r < count; r++ )
			{
				bvh.Intersect( ref working[ r ] );
				if ( working[ r ].Hit.T < BvhConstants.Far )
				{
					hits++;
				}
			}
			return ( double )hits / count;
		}

		static double Median( double[] values )
		{
			double[] sorted = ( double[] )values.Clone();
			Array.Sort( sorted );
			int mid = sorted.Length / 2;
			return sorted.Length % 2 == 0 ? ( sorted[ mid - 1 ] + sorted[ mid ] ) * 0.5 : sorted[ mid ];
		}

		static double MraysPerSecond( int rayCount, double milliseconds )
		{
			return rayCount / ( milliseconds * 1000.0 );
		}

		static string FormatMs( double milliseconds )
		{
			return milliseconds < 100.0 ? milliseconds.ToString( "F1", CultureInfo.InvariantCulture ) : milliseconds.ToString( "F0", CultureInfo.InvariantCulture );
		}

		static string FormatMrays( double mraysPerSecond )
		{
			return mraysPerSecond.ToString( "F2", CultureInfo.InvariantCulture );
		}

		static string FormatMachineLine()
		{
			string bitness = Environment.Is64BitProcess ? "64-bit" : "32-bit";
			return $"{RuntimeInformation.FrameworkDescription}, {RuntimeInformation.OSDescription}, {RuntimeInformation.ProcessArchitecture}, {Environment.ProcessorCount} processors, {bitness}";
		}
	}
}
