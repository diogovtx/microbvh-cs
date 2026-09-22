// refdump.cpp - generates deterministic reference data from tinybvh (v1.8.0)
// for validating microBVH.
//
// Usage: refdump <scene.bin> <out.ref>
//
// Building: download tiny_bvh.h as of tinybvh's 1.8.0 release from
//   https://raw.githubusercontent.com/jbikker/tinybvh/0e4584287823252cf83f0e9cd072848bec5f79c5/tiny_bvh.h
// and save it next to this file, then compile, e.g. with MSVC:
//   cl /O2 /EHsc /std:c++20 /fp:precise refdump.cpp
// With GCC or Clang, add -ffp-contract=off: fused multiply-adds change the results in the last
// bits, and microBVH does not fuse them.
//
// Scene file format (input):
//   int32 triCount
//   triCount*3 vertices, each a 16-byte float4 (x,y,z,w); three consecutive
//   vertices form a triangle.
//
// Output file format (little-endian, no padding):
//   char[8]  magic = "MBVHREF1"
//   u32      triCount
//   -- BLAS section --
//   u32      usedNodes
//   f32      sahCost
//   f32[3]   aabbMin, f32[3] aabbMax            (root bounds)
//   u32      nodeCount (== usedNodes)
//            nodeCount * 32 bytes: raw BVH::BVHNode (aabbMin xyz, leftFirst, aabbMax xyz, triCount)
//   u32      idxCount
//            idxCount * u32: bvh.primIdx[]
//   -- BLAS rays --
//   u32      rayCount (65536)
//            per ray: f32[3] O, f32[3] D, f32 t, f32 u, f32 v, u32 prim, u32 occludedFull, u32 occludedHalf
//            (9 floats + 3 u32 = 48 bytes per ray)
//   -- Refit section --
//   f32[3]   aabbMin, f32[3] aabbMax            (root bounds after refit)
//            per ray (same count/order as BLAS rays): f32 t, f32 u, f32 v, u32 prim   (16 bytes)
//   -- TLAS section --
//   u32      instCount (3)
//            per instance: f32[16] transform (cell[0..15] row-major), f32[16] invTransform,
//                           f32[3] aabbMin, f32[3] aabbMax, u32 mask
//   u32      tlasUsedNodes
//   f32[3]   tlasAabbMin, f32[3] tlasAabbMax
//   u32      tlasNodeCount; tlasNodeCount * 32 bytes raw nodes
//   u32      tlasIdxCount;  tlasIdxCount * u32 primIdx
//   u32      tlasRayCount (65536)
//            per ray: f32[3] O, f32[3] D, f32 t, f32 u, f32 v, u32 prim, u32 inst, u32 occludedFull
//            (9 floats + 3 u32 = 48 bytes per ray)
//   -- Indexed-geometry section --
//   Everything below validates the indexed build path, i.e. BVH::Build( slice, indices, prims )
//   and the vertIdx / GET_PRIM_INDICES_I0_I1_I2 addressing that hangs off it. The triangle soup
//   is welded into an indexed mesh here so the consumer does not have to reproduce the welding:
//   vertices are matched on the exact bit pattern of x, y and z (w is carried over from the first
//   occurrence and never compared), the first occurrence of a position wins, and the soup is
//   walked in order, so the welded array and the index array are bit-reproducible. The welded
//   vertex count is generally not triCount*3, so the build below passes an explicit
//   bvhvec4slice with the real count: the bvhvec4* convenience overload of Build hardcodes
//   count = prims*3, which is wrong for an indexed mesh.
//   u32      weldedVertCount
//            weldedVertCount * 16 bytes: the welded vertices, as float4
//   u32      indexCount (== triCount * 3)
//            indexCount * u32: the index array, three per triangle, in triangle order
//   then the binned SAH build over the indexed mesh (BVH::Build):
//   u32      idxUsedNodes
//   u32      idxNodeCount (== idxUsedNodes)
//            idxNodeCount * 32 bytes: raw BVH::BVHNode, same layout as the BLAS nodes
//   u32      idxPrimCount
//            idxPrimCount * u32: bvh.primIdx[]   (idxCount; == triCount for a binned build)
//   u32      idxRayCount (65536)
//            the same rays as the BLAS ray section, in the same order, traced against the
//            indexed tree. Identical 48-byte record to the BLAS rays.
//
// Determinism strategy: NO_THREADED_BUILDS is defined before including
// tiny_bvh.h. This compiles ENABLE_THREADED_BUILDS out entirely, so every
// `#ifdef ENABLE_THREADED_BUILDS` branch that spawns worker tasks during a
// build is removed at compile time and `threadedBuild` is unconditionally
// forced to false right before those branches. TINYBVH_NO_SIMD is defined so
// BVH_USEAVX/BVH_USENEON never get set, which routes BVH::Build(verts,triCount)
// through PrepareBuild()+Build() - the scalar reference builder - rather than
// BuildAVX(). bvh.settings.useSIMDifavailable is also set to false, and
// bvh.context.spawn/barrier/parallel_for are nulled explicitly, for belt and
// braces on top of the compile-time switch.

#define TINYBVH_NO_SIMD
#define NO_THREADED_BUILDS
#define TINYBVH_IMPLEMENTATION
#include "tiny_bvh.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <fstream>
#include <array>
#include <map>

using namespace tinybvh;

static const uint32_t N_RAYS = 65536;
static const float PI = 3.14159265358979323846f;

// Fixed LCG, as specified - do not use rand().
static uint32_t s = 0x12345678;
static float R()
{
	s = s * 1664525u + 1013904223u;
	return (s >> 8) * (1.0f / 16777216.0f);
}

static bvhvec3 RandomUnitVector()
{
	float z = 2.0f * R() - 1.0f;
	float phi = 2.0f * PI * R();
	float r = sqrtf( 1.0f - z * z );
	return bvhvec3( r * cosf( phi ), r * sinf( phi ), z );
}

static bvhvec3 RandomPointInAABB( const bvhvec3& aabbMin, const bvhvec3& aabbMax )
{
	return bvhvec3(
		aabbMin.x + R() * (aabbMax.x - aabbMin.x),
		aabbMin.y + R() * (aabbMax.y - aabbMin.y),
		aabbMin.z + R() * (aabbMax.z - aabbMin.z) );
}

// One generated ray, kept around so it can be replayed against the refit BVH
// without needing to re-run the RNG.
struct GenRay { bvhvec3 O, D; };

static std::vector<GenRay> GenerateRays( const bvhvec3& aabbMin, const bvhvec3& aabbMax, uint32_t count )
{
	const bvhvec3 center = (aabbMin + aabbMax) * 0.5f;
	const float radius = tinybvh_length( aabbMax - aabbMin ) * 0.5f;
	std::vector<GenRay> rays( count );
	for (uint32_t i = 0; i < count; i++)
	{
		bvhvec3 origin = center + RandomUnitVector() * radius * 1.2f;
		bvhvec3 target = RandomPointInAABB( aabbMin, aabbMax );
		Ray ray( origin, target - origin ); // constructor normalizes D and computes rD
		rays[i].O = ray.O, rays[i].D = ray.D;
	}
	return rays;
}

static void WriteVec3( std::ofstream& f, const bvhvec3& v )
{
	f.write( (const char*)&v.x, 4 );
	f.write( (const char*)&v.y, 4 );
	f.write( (const char*)&v.z, 4 );
}

// One traced BLAS-style ray record; 48 bytes once the ray's O and D are written in front of it.
struct BlasHit { float t, u, v; uint32_t prim, occludedFull, occludedHalf; };

static void TraceBlasRays( BVH& bvh, const std::vector<GenRay>& rays, std::vector<BlasHit>& out )
{
	out.resize( rays.size() );
	for (size_t i = 0; i < rays.size(); i++)
	{
		const GenRay& gr = rays[i];
		Ray ray( gr.O, gr.D );
		bvh.Intersect( ray );
		BlasHit& h = out[i];
		h.t = ray.hit.t, h.u = ray.hit.u, h.v = ray.hit.v, h.prim = ray.hit.prim;

		Ray occRay( gr.O, gr.D );
		h.occludedFull = bvh.IsOccluded( occRay ) ? 1u : 0u;

		float halfT = ray.hit.t < BVH_FAR ? 0.5f * ray.hit.t : BVH_FAR;
		Ray occHalfRay( gr.O, gr.D, halfT );
		h.occludedHalf = bvh.IsOccluded( occHalfRay ) ? 1u : 0u;
	}
}

static void WriteU32( std::ofstream& f, uint32_t v ) { f.write( (const char*)&v, 4 ); }
static void WriteF32( std::ofstream& f, float v ) { f.write( (const char*)&v, 4 ); }

// Writes the raw node pool and primitive-index array for a built BVH (BLAS or TLAS).
static void WriteNodesAndIndices( std::ofstream& f, const BVH& bvh )
{
	WriteU32( f, bvh.usedNodes );
	f.write( (const char*)bvh.bvhNode, (size_t)bvh.usedNodes * sizeof( BVH::BVHNode ) );
	WriteU32( f, bvh.idxCount );
	f.write( (const char*)bvh.primIdx, (size_t)bvh.idxCount * sizeof( uint32_t ) );
}

// Writes the ray block of the indexed-geometry section: the count, then a 48-byte record per
// ray holding its O and D followed by the hit it produced.
static void WriteRayRecords( std::ofstream& f, const std::vector<GenRay>& rays, const std::vector<BlasHit>& hits )
{
	WriteU32( f, (uint32_t)rays.size() );
	for (size_t i = 0; i < rays.size(); i++)
	{
		WriteVec3( f, rays[i].O ); WriteVec3( f, rays[i].D );
		const BlasHit& h = hits[i];
		WriteF32( f, h.t ); WriteF32( f, h.u ); WriteF32( f, h.v );
		WriteU32( f, h.prim ); WriteU32( f, h.occludedFull ); WriteU32( f, h.occludedHalf );
	}
}

int main( int argc, char** argv )
{
	static_assert (sizeof( BVH::BVHNode ) == 32, "BVH::BVHNode must be 32 bytes");

	if (argc != 3)
	{
		fprintf( stderr, "usage: refdump <scene.bin> <out.ref>\n" );
		return 1;
	}
	const char* sceneFile = argv[1];
	const char* outFile = argv[2];

	// --- Load scene -----------------------------------------------------
	std::ifstream sf( sceneFile, std::ios::binary );
	if (!sf)
	{
		fprintf( stderr, "cannot open scene file: %s\n", sceneFile );
		return 1;
	}
	int32_t triCount = 0;
	sf.read( (char*)&triCount, 4 );
	const uint32_t vertCount = (uint32_t)triCount * 3;
	bvhvec4* verts = (bvhvec4*)malloc64( vertCount * sizeof( bvhvec4 ) );
	sf.read( (char*)verts, (size_t)vertCount * sizeof( bvhvec4 ) );
	sf.close();

	// keep a pristine copy of the vertices, for the refit test.
	std::vector<bvhvec4> vertsOriginal( verts, verts + vertCount );

	// --- Determinism check: build the same scene twice, compare byte-for-byte.
	BVH bvhA, bvhB;
	bvhA.settings.useSIMDifavailable = false;
	bvhA.context.spawn = nullptr, bvhA.context.barrier = nullptr, bvhA.context.parallel_for = nullptr;
	bvhB.settings.useSIMDifavailable = false;
	bvhB.context.spawn = nullptr, bvhB.context.barrier = nullptr, bvhB.context.parallel_for = nullptr;
	bvhA.Build( verts, (uint32_t)triCount );
	bvhB.Build( verts, (uint32_t)triCount );

	bool deterministic = bvhA.usedNodes == bvhB.usedNodes && bvhA.idxCount == bvhB.idxCount;
	if (deterministic)
	{
		deterministic = memcmp( bvhA.bvhNode, bvhB.bvhNode, (size_t)bvhA.usedNodes * sizeof( BVH::BVHNode ) ) == 0
			&& memcmp( bvhA.primIdx, bvhB.primIdx, (size_t)bvhA.idxCount * sizeof( uint32_t ) ) == 0;
	}
	if (!deterministic)
	{
		fprintf( stderr, "FATAL: two builds of the same scene produced different BVHs - not deterministic.\n" );
		return 1;
	}
	if (bvhA.may_have_holes)
	{
		fprintf( stderr, "FATAL: bvh.may_have_holes is true; expected a hole-free serial build.\n" );
		return 1;
	}

	// bvhA is the working BLAS from here on; bvhB was only needed for the check above.
	BVH& bvh = bvhA;

	// --- BLAS section -----------------------------------------------------
	const float sahCost = bvh.SAHCost();
	const bvhvec3 blasAabbMin = bvh.aabbMin, blasAabbMax = bvh.aabbMax;

	// snapshot the node/index arrays before Refit() touches them.
	std::vector<BVH::BVHNode> blasNodes( bvh.bvhNode, bvh.bvhNode + bvh.usedNodes );
	std::vector<uint32_t> blasIdx( bvh.primIdx, bvh.primIdx + bvh.idxCount );

	// --- BLAS rays ----------------------------------------------------
	std::vector<GenRay> blasRays = GenerateRays( blasAabbMin, blasAabbMax, N_RAYS );
	std::vector<BlasHit> blasHits( N_RAYS );
	for (uint32_t i = 0; i < N_RAYS; i++)
	{
		const GenRay& gr = blasRays[i];
		Ray ray( gr.O, gr.D );
		bvh.Intersect( ray );
		BlasHit& h = blasHits[i];
		h.t = ray.hit.t, h.u = ray.hit.u, h.v = ray.hit.v, h.prim = ray.hit.prim;

		Ray occRay( gr.O, gr.D );
		h.occludedFull = bvh.IsOccluded( occRay ) ? 1u : 0u;

		float halfT = ray.hit.t < BVH_FAR ? 0.5f * ray.hit.t : BVH_FAR;
		Ray occHalfRay( gr.O, gr.D, halfT );
		h.occludedHalf = bvh.IsOccluded( occHalfRay ) ? 1u : 0u;
	}

	// --- Indexed-geometry section ----------------------------------------
	// Built here, while the vertices are still pristine. See the format comment at the top of
	// this file for the welding rule; the key is the raw bit pattern of x, y and z, and std::map
	// is used rather than a hash map so nothing about the result can depend on bucket layout.
	std::vector<bvhvec4> welded;
	std::vector<uint32_t> indices( vertCount );
	welded.reserve( vertCount );
	{
		std::map<std::array<uint32_t, 3>, uint32_t> seen;
		for (uint32_t i = 0; i < vertCount; i++)
		{
			std::array<uint32_t, 3> key;
			memcpy( key.data(), &verts[i].x, 3 * sizeof( uint32_t ) );
			std::map<std::array<uint32_t, 3>, uint32_t>::const_iterator it = seen.find( key );
			if (it == seen.end())
			{
				const uint32_t slot = (uint32_t)welded.size();
				seen.insert( std::make_pair( key, slot ) );
				welded.push_back( verts[i] );
				indices[i] = slot;
			}
			else indices[i] = it->second;
		}
	}
	const uint32_t weldedCount = (uint32_t)welded.size();
	// give the welded vertices the same 64-byte aligned storage the soup gets.
	bvhvec4* weldedVerts = (bvhvec4*)malloc64( weldedCount * sizeof( bvhvec4 ) );
	memcpy( weldedVerts, welded.data(), (size_t)weldedCount * sizeof( bvhvec4 ) );
	const bvhvec4slice weldedSlice( weldedVerts, weldedCount, sizeof( bvhvec4 ) );

	BVH ibvh;
	ibvh.settings.useSIMDifavailable = false;
	ibvh.context.spawn = nullptr, ibvh.context.barrier = nullptr, ibvh.context.parallel_for = nullptr;
	ibvh.Build( weldedSlice, indices.data(), (uint32_t)triCount );
	std::vector<BlasHit> idxHits;
	TraceBlasRays( ibvh, blasRays, idxHits );

	// --- Refit section ------------------------------------------------
	// deterministic per-vertex perturbation.
	const bvhvec3 ext = blasAabbMax - blasAabbMin;
	for (uint32_t i = 0; i < vertCount; i++)
	{
		bvhvec4& v = verts[i];
		v.x = v.x + (float)((int)(i % 7) - 3) * 0.005f * ext.x;
		v.y = v.y + (float)((int)(i % 5) - 2) * 0.005f * ext.y;
	}
	bvh.Refit();
	const bvhvec3 refitAabbMin = bvh.aabbMin, refitAabbMax = bvh.aabbMax;

	struct RefitHit { float t, u, v; uint32_t prim; };
	std::vector<RefitHit> refitHits( N_RAYS );
	for (uint32_t i = 0; i < N_RAYS; i++)
	{
		const GenRay& gr = blasRays[i];
		Ray ray( gr.O, gr.D );
		bvh.Intersect( ray );
		refitHits[i] = { ray.hit.t, ray.hit.u, ray.hit.v, ray.hit.prim };
	}

	// restore original vertices for the TLAS test.
	memcpy( verts, vertsOriginal.data(), (size_t)vertCount * sizeof( bvhvec4 ) );
	bvh.Refit(); // restore node bounds to match the un-perturbed geometry, for the TLAS test

	// --- TLAS section ---------------------------------------------------
	BLASInstance inst[3] = { BLASInstance( 0 ), BLASInstance( 0 ), BLASInstance( 0 ) };
	inst[0].mask = 0xFFFF; // identity, default transform

	inst[1].transform.cell[3] = ext.x * 1.1f;
	inst[1].mask = 0xFFFF;

	inst[2].transform.cell[0] = 0, inst[2].transform.cell[1] = 0, inst[2].transform.cell[2] = 0.5f, inst[2].transform.cell[3] = 0;
	inst[2].transform.cell[4] = 0, inst[2].transform.cell[5] = 0.5f, inst[2].transform.cell[6] = 0, inst[2].transform.cell[7] = 0;
	inst[2].transform.cell[8] = -0.5f, inst[2].transform.cell[9] = 0, inst[2].transform.cell[10] = 0, inst[2].transform.cell[11] = ext.z * 1.1f;
	inst[2].mask = 0x0002;

	BVH tlas;
	tlas.settings.useSIMDifavailable = false;
	tlas.context.spawn = nullptr, tlas.context.barrier = nullptr, tlas.context.parallel_for = nullptr;
	BVHBase* blasList[1] = { &bvh };
	tlas.Build( inst, 3, blasList, 1 );

	std::vector<GenRay> tlasRays = GenerateRays( tlas.aabbMin, tlas.aabbMax, N_RAYS );
	struct TlasHit { float t, u, v; uint32_t prim, inst, occludedFull; };
	std::vector<TlasHit> tlasHits( N_RAYS );
	for (uint32_t i = 0; i < N_RAYS; i++)
	{
		const GenRay& gr = tlasRays[i];
		Ray ray( gr.O, gr.D );
		tlas.Intersect( ray );
		TlasHit& h = tlasHits[i];
		h.t = ray.hit.t, h.u = ray.hit.u, h.v = ray.hit.v, h.prim = ray.hit.prim, h.inst = ray.hit.inst;

		Ray occRay( gr.O, gr.D );
		h.occludedFull = tlas.IsOccluded( occRay ) ? 1u : 0u;
	}

	// --- Write output file ----------------------------------------------
	std::ofstream f( outFile, std::ios::binary );
	if (!f)
	{
		fprintf( stderr, "cannot open output file: %s\n", outFile );
		return 1;
	}
	f.write( "MBVHREF1", 8 );
	WriteU32( f, (uint32_t)triCount );

	// BLAS section
	WriteU32( f, bvh.usedNodes );
	WriteF32( f, sahCost );
	WriteVec3( f, blasAabbMin ); WriteVec3( f, blasAabbMax );
	WriteU32( f, (uint32_t)blasNodes.size() );
	f.write( (const char*)blasNodes.data(), blasNodes.size() * sizeof( BVH::BVHNode ) );
	WriteU32( f, (uint32_t)blasIdx.size() );
	f.write( (const char*)blasIdx.data(), blasIdx.size() * sizeof( uint32_t ) );

	// BLAS rays
	WriteU32( f, N_RAYS );
	for (uint32_t i = 0; i < N_RAYS; i++)
	{
		WriteVec3( f, blasRays[i].O ); WriteVec3( f, blasRays[i].D );
		const BlasHit& h = blasHits[i];
		WriteF32( f, h.t ); WriteF32( f, h.u ); WriteF32( f, h.v );
		WriteU32( f, h.prim ); WriteU32( f, h.occludedFull ); WriteU32( f, h.occludedHalf );
	}

	// Refit section
	WriteVec3( f, refitAabbMin ); WriteVec3( f, refitAabbMax );
	for (uint32_t i = 0; i < N_RAYS; i++)
	{
		const RefitHit& h = refitHits[i];
		WriteF32( f, h.t ); WriteF32( f, h.u ); WriteF32( f, h.v ); WriteU32( f, h.prim );
	}

	// TLAS section
	WriteU32( f, 3 );
	for (int i = 0; i < 3; i++)
	{
		f.write( (const char*)inst[i].transform.cell, 16 * sizeof( float ) );
		f.write( (const char*)inst[i].invTransform.cell, 16 * sizeof( float ) );
		WriteVec3( f, inst[i].aabbMin ); WriteVec3( f, inst[i].aabbMax );
		WriteU32( f, inst[i].mask );
	}
	WriteU32( f, tlas.usedNodes );
	WriteVec3( f, tlas.aabbMin ); WriteVec3( f, tlas.aabbMax );
	WriteNodesAndIndices( f, tlas ); // writes tlasNodeCount/nodes then tlasIdxCount/idx
	WriteU32( f, N_RAYS );
	for (uint32_t i = 0; i < N_RAYS; i++)
	{
		WriteVec3( f, tlasRays[i].O ); WriteVec3( f, tlasRays[i].D );
		const TlasHit& h = tlasHits[i];
		WriteF32( f, h.t ); WriteF32( f, h.u ); WriteF32( f, h.v );
		WriteU32( f, h.prim ); WriteU32( f, h.inst ); WriteU32( f, h.occludedFull );
	}

	// Indexed-geometry section
	WriteU32( f, weldedCount );
	f.write( (const char*)weldedVerts, (size_t)weldedCount * sizeof( bvhvec4 ) );
	WriteU32( f, (uint32_t)indices.size() );
	f.write( (const char*)indices.data(), indices.size() * sizeof( uint32_t ) );
	WriteU32( f, ibvh.usedNodes );
	WriteU32( f, ibvh.usedNodes );
	f.write( (const char*)ibvh.bvhNode, (size_t)ibvh.usedNodes * sizeof( BVH::BVHNode ) );
	WriteU32( f, ibvh.idxCount );
	f.write( (const char*)ibvh.primIdx, (size_t)ibvh.idxCount * sizeof( uint32_t ) );
	WriteRayRecords( f, blasRays, idxHits );
	f.close();

	// --- Summary ----------------------------------------------------------
	uint32_t blasHitCount = 0, refitHitCount = 0, tlasHitCount = 0;
	for (uint32_t i = 0; i < N_RAYS; i++) if (blasHits[i].t < BVH_FAR) blasHitCount++;
	for (uint32_t i = 0; i < N_RAYS; i++) if (refitHits[i].t < BVH_FAR) refitHitCount++;
	for (uint32_t i = 0; i < N_RAYS; i++) if (tlasHits[i].t < BVH_FAR) tlasHitCount++;

	printf( "scene: %s\n", sceneFile );
	printf( "  triCount:       %d\n", triCount );
	printf( "  usedNodes:      %u\n", bvh.usedNodes );
	printf( "  SAH cost:       %f\n", sahCost );
	printf( "  BLAS hit ratio:  %.2f%% (%u/%u)\n", 100.0 * blasHitCount / N_RAYS, blasHitCount, N_RAYS );
	printf( "  refit hit ratio: %.2f%% (%u/%u)\n", 100.0 * refitHitCount / N_RAYS, refitHitCount, N_RAYS );
	printf( "  TLAS hit ratio:  %.2f%% (%u/%u)\n", 100.0 * tlasHitCount / N_RAYS, tlasHitCount, N_RAYS );
	uint32_t idxHitCount = 0;
	for (uint32_t i = 0; i < N_RAYS; i++) if (idxHits[i].t < BVH_FAR) idxHitCount++;
	printf( "  indexed: welded %u verts from %u (%.2fx), nodes %u\n",
		weldedCount, vertCount, (double)weldedCount / vertCount, ibvh.usedNodes );
	printf( "  indexed hit ratio: %.2f%% (%u/%u)\n", 100.0 * idxHitCount / N_RAYS, idxHitCount, N_RAYS );
	printf( "  determinism check: PASS (two independent builds are byte-identical)\n" );
	printf( "  may_have_holes: %s\n", bvh.may_have_holes ? "true (UNEXPECTED)" : "false" );
	printf( "  wrote: %s\n", outFile );

	free64( verts );
	free64( weldedVerts );
	return 0;
}
