using System.IO;
using System.Text;

namespace MicroBVH.Tests
{
	/// <summary>
	/// Reads the ".ref" reference file produced by Tools/RefDump/refdump.cpp (format "MBVHREF1").
	/// See that file's header comment for the authoritative layout; this reader matches it exactly,
	/// including the 48-byte per-ray records for the BLAS and TLAS ray sections.
	/// </summary>
	public class RefDumpFile
	{
		public struct RayHit
		{
			public BvhVec3 O;
			public BvhVec3 D;
			public float T;
			public float U;
			public float V;
			public uint Prim;
			public uint OccludedFull;
			public uint OccludedHalf;
		}

		public struct RefitHit
		{
			public float T;
			public float U;
			public float V;
			public uint Prim;
		}

		public struct TlasRayHit
		{
			public BvhVec3 O;
			public BvhVec3 D;
			public float T;
			public float U;
			public float V;
			public uint Prim;
			public uint Inst;
			public uint OccludedFull;
		}

		public struct InstanceRecord
		{
			public BvhMat4 Transform;
			public BvhMat4 InvTransform;
			public BvhVec3 AabbMin;
			public BvhVec3 AabbMax;
			public uint Mask;
		}

		public uint TriCount;

		// BLAS section.
		public uint UsedNodes;
		public float SahCost;
		public BvhVec3 AabbMin;
		public BvhVec3 AabbMax;
		public BvhNode[] Nodes;
		public uint[] PrimIdx;

		// BLAS rays.
		public RayHit[] Rays;

		// Refit section.
		public BvhVec3 RefitAabbMin;
		public BvhVec3 RefitAabbMax;
		public RefitHit[] RefitHits;

		// TLAS section.
		public InstanceRecord[] Instances;
		public uint TlasUsedNodes;
		public BvhVec3 TlasAabbMin;
		public BvhVec3 TlasAabbMax;
		public BvhNode[] TlasNodes;
		public uint[] TlasPrimIdx;
		public TlasRayHit[] TlasRays;

		// Indexed-geometry section: the same triangles welded into an indexed mesh, plus a binned
		// build over it. The welding was done by the dump tool; read these arrays rather than
		// re-welding, so the two sides cannot drift apart.
		public BvhVec4[] WeldedVertices;
		/// <summary>Three vertex indices per triangle, in triangle order; length is TriCount * 3.</summary>
		public uint[] Indices;
		public uint IndexedUsedNodes;
		public BvhNode[] IndexedNodes;
		public uint[] IndexedPrimIdx;
		public RayHit[] IndexedRays;

		public static RefDumpFile Load( string path )
		{
			// Buffered: these dumps can be up to ~35 MB; BufferedStream cuts down the syscall count
			// behind BinaryReader's many small field-at-a-time reads.
			using ( BufferedStream buffered = new BufferedStream( File.OpenRead( path ), 1 << 20 ) )
			using ( BinaryReader reader = new BinaryReader( buffered ) )
			{
				byte[] magic = reader.ReadBytes( 8 );
				string magicStr = Encoding.ASCII.GetString( magic );
				if ( magicStr != "MBVHREF1" )
				{
					throw new InvalidDataException( "unexpected magic: " + magicStr );
				}

				RefDumpFile file = new RefDumpFile();
				file.TriCount = reader.ReadUInt32();

				// BLAS section.
				file.UsedNodes = reader.ReadUInt32();
				file.SahCost = reader.ReadSingle();
				file.AabbMin = ReadFloat3( reader );
				file.AabbMax = ReadFloat3( reader );
				uint nodeCount = reader.ReadUInt32();
				file.Nodes = ReadNodes( reader, nodeCount );
				uint idxCount = reader.ReadUInt32();
				file.PrimIdx = ReadUInts( reader, idxCount );

				// BLAS rays.
				uint rayCount = reader.ReadUInt32();
				file.Rays = new RayHit[ rayCount ];
				for ( uint i = 0; i < rayCount; i++ )
				{
					RayHit hit;
					hit.O = ReadFloat3( reader );
					hit.D = ReadFloat3( reader );
					hit.T = reader.ReadSingle();
					hit.U = reader.ReadSingle();
					hit.V = reader.ReadSingle();
					hit.Prim = reader.ReadUInt32();
					hit.OccludedFull = reader.ReadUInt32();
					hit.OccludedHalf = reader.ReadUInt32();
					file.Rays[ i ] = hit;
				}

				// Refit section: root bounds, then rayCount hits in the same order as the BLAS rays.
				file.RefitAabbMin = ReadFloat3( reader );
				file.RefitAabbMax = ReadFloat3( reader );
				file.RefitHits = new RefitHit[ rayCount ];
				for ( uint i = 0; i < rayCount; i++ )
				{
					RefitHit hit;
					hit.T = reader.ReadSingle();
					hit.U = reader.ReadSingle();
					hit.V = reader.ReadSingle();
					hit.Prim = reader.ReadUInt32();
					file.RefitHits[ i ] = hit;
				}

				// TLAS section.
				uint instCount = reader.ReadUInt32();
				file.Instances = new InstanceRecord[ instCount ];
				for ( uint i = 0; i < instCount; i++ )
				{
					InstanceRecord inst;
					inst.Transform = ReadMat4( reader );
					inst.InvTransform = ReadMat4( reader );
					inst.AabbMin = ReadFloat3( reader );
					inst.AabbMax = ReadFloat3( reader );
					inst.Mask = reader.ReadUInt32();
					file.Instances[ i ] = inst;
				}

				file.TlasUsedNodes = reader.ReadUInt32();
				file.TlasAabbMin = ReadFloat3( reader );
				file.TlasAabbMax = ReadFloat3( reader );
				uint tlasNodeCount = reader.ReadUInt32();
				file.TlasNodes = ReadNodes( reader, tlasNodeCount );
				uint tlasIdxCount = reader.ReadUInt32();
				file.TlasPrimIdx = ReadUInts( reader, tlasIdxCount );

				uint tlasRayCount = reader.ReadUInt32();
				file.TlasRays = new TlasRayHit[ tlasRayCount ];
				for ( uint i = 0; i < tlasRayCount; i++ )
				{
					TlasRayHit hit;
					hit.O = ReadFloat3( reader );
					hit.D = ReadFloat3( reader );
					hit.T = reader.ReadSingle();
					hit.U = reader.ReadSingle();
					hit.V = reader.ReadSingle();
					hit.Prim = reader.ReadUInt32();
					hit.Inst = reader.ReadUInt32();
					hit.OccludedFull = reader.ReadUInt32();
					file.TlasRays[ i ] = hit;
				}

				// Indexed-geometry section: the welded mesh, then a binned build over it, followed
				// by the BLAS rays traced against it.
				uint weldedCount = reader.ReadUInt32();
				file.WeldedVertices = new BvhVec4[ weldedCount ];
				for ( uint i = 0; i < weldedCount; i++ )
				{
					float x = reader.ReadSingle();
					float y = reader.ReadSingle();
					float z = reader.ReadSingle();
					float w = reader.ReadSingle();
					file.WeldedVertices[ i ] = new BvhVec4( x, y, z, w );
				}
				uint indexCount = reader.ReadUInt32();
				file.Indices = ReadUInts( reader, indexCount );

				file.IndexedUsedNodes = reader.ReadUInt32();
				uint indexedNodeCount = reader.ReadUInt32();
				file.IndexedNodes = ReadNodes( reader, indexedNodeCount );
				uint indexedPrimCount = reader.ReadUInt32();
				file.IndexedPrimIdx = ReadUInts( reader, indexedPrimCount );
				file.IndexedRays = ReadRays( reader );

				if ( buffered.Position != buffered.Length )
				{
					throw new InvalidDataException( "trailing data in " + path );
				}
				return file;
			}
		}

		/// <summary>Reads one count-prefixed block of 48-byte ray records.</summary>
		internal static RayHit[] ReadRays( BinaryReader reader )
		{
			uint rayCount = reader.ReadUInt32();
			RayHit[] rays = new RayHit[ rayCount ];
			for ( uint i = 0; i < rayCount; i++ )
			{
				RayHit hit;
				hit.O = ReadFloat3( reader );
				hit.D = ReadFloat3( reader );
				hit.T = reader.ReadSingle();
				hit.U = reader.ReadSingle();
				hit.V = reader.ReadSingle();
				hit.Prim = reader.ReadUInt32();
				hit.OccludedFull = reader.ReadUInt32();
				hit.OccludedHalf = reader.ReadUInt32();
				rays[ i ] = hit;
			}
			return rays;
		}

		internal static BvhVec3 ReadFloat3( BinaryReader reader )
		{
			float x = reader.ReadSingle();
			float y = reader.ReadSingle();
			float z = reader.ReadSingle();
			return new BvhVec3( x, y, z );
		}

		internal static BvhMat4 ReadMat4( BinaryReader reader )
		{
			BvhMat4 m = default;
			for ( int i = 0; i < 16; i++ )
			{
				m[ i ] = reader.ReadSingle();
			}
			return m;
		}

		static BvhNode ReadNode( BinaryReader reader )
		{
			BvhNode node;
			node.AabbMin = ReadFloat3( reader );
			node.LeftFirst = reader.ReadUInt32();
			node.AabbMax = ReadFloat3( reader );
			node.TriCount = reader.ReadUInt32();
			return node;
		}

		internal static BvhNode[] ReadNodes( BinaryReader reader, uint count )
		{
			BvhNode[] nodes = new BvhNode[ count ];
			for ( uint i = 0; i < count; i++ )
			{
				nodes[ i ] = ReadNode( reader );
			}
			return nodes;
		}

		internal static uint[] ReadUInts( BinaryReader reader, uint count )
		{
			uint[] values = new uint[ count ];
			for ( uint i = 0; i < count; i++ )
			{
				values[ i ] = reader.ReadUInt32();
			}
			return values;
		}
	}
}
