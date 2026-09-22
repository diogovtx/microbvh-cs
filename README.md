# microBVH

A small, portable, single-file C# subset of [tinybvh](https://github.com/jbikker/tinybvh) 1.8.0:
bounding volume hierarchy construction and ray traversal in about 2,100 lines of safe C#.

- One file, `MicroBVH.cs`. Copy it into any project that targets .NET Standard 2.1 or later:
  .NET Core 3+ and .NET 5+, Mono, Unity 2021.2+ and Godot 4 all qualify.
- Safe C# 9 only: no `unsafe`, no pointers, no SIMD intrinsics, nothing beyond the base class
  library. It also works in projects built with overflow checking.
- Same trees and same hits as tinybvh's scalar build, bit for bit, validated against reference
  data produced by the C++ library.

## What it does

| | |
| --- | --- |
| Build | Binned SAH, tinybvh's reference builder, over a triangle soup `Build( vertices, triCount )` or an indexed mesh `Build( vertices, indices, triCount )`. Set `UseThreadedBuild` to build inputs of 50,000 primitives or more on the thread pool. |
| Custom geometry | `BuildAabbs( aabbs, count )` or `Build( getAabb, count )`, traced through the `CustomIntersect` / `CustomIsOccluded` callbacks. |
| Instancing | `BuildTlas( instances, count, blasses, blasCount )`: a top-level BVH over `BlasInstance`s with 4x4 transforms and 16-bit ray masks. |
| Traversal | `Intersect( ref ray )` for the closest hit, `IsOccluded( ray )` for any hit. Safe to call from many threads at once. |
| Maintenance | `Refit()` after moving vertices; `SahCost()`, `NodeCount()`, `PrimCount()` and `LeafCount()` to inspect a tree. |

Not included: the SBVH and the other alternative builders, the tree optimizer, the wide CPU and
GPU layouts, SIMD traversal, double precision, voxel sets, opacity micromaps, ray packets, sphere
queries and saving trees to disk. tinybvh has all of those.

## Usage

```csharp
using MicroBVH;

// three vertices per triangle; w is not used. The BVH references this array: keep it alive.
BvhVec4[] vertices = LoadTriangles( out uint triCount );
Bvh bvh = new Bvh();
bvh.Build( vertices, triCount );

// closest hit; the direction is normalized for you.
Ray ray = new Ray( new BvhVec3( 0f, 1f, -5f ), new BvhVec3( 0f, 0f, 1f ) );
bvh.Intersect( ref ray );
if ( ray.Hit.T < BvhConstants.Far )
{
	// ray.Hit.Prim is the triangle, ray.Hit.U / ray.Hit.V its barycentrics.
}

// any hit within a distance, e.g. a shadow ray.
bool shadowed = bvh.IsOccluded( new Ray( point, toLight, distanceToLight ) );

// same triangles, new positions: refit instead of rebuilding.
bvh.Refit();

// instancing: two copies of the same mesh, the second moved 10 units along x.
BlasInstance[] instances = { BlasInstance.Create( 0 ), BlasInstance.Create( 0 ) };
instances[ 1 ].Transform[ 3 ] = 10f; // cells are row-major, so 3, 7 and 11 hold the translation
Bvh tlas = new Bvh();
tlas.BuildTlas( instances, 2, new[] { bvh }, 1 );
tlas.Intersect( ref ray ); // ray.Hit.Inst tells which instance was hit
```

Custom primitives supply their bounds and an intersection test:

```csharp
Bvh spheres = new Bvh();
spheres.Build( ( uint i, out BvhVec3 min, out BvhVec3 max ) =>
{
	min = centers[ i ] - new BvhVec3( radius );
	max = centers[ i ] + new BvhVec3( radius );
}, sphereCount );
spheres.CustomIntersect = ( ref Ray r, uint i ) => IntersectSphere( ref r, i ); // updates r.Hit when closer
spheres.CustomIsOccluded = ( in Ray r, uint i ) => SphereBlocks( r, i );
```

## Tests

`Tests/` holds NUnit tests that compare microBVH with reference dumps of the C++ library on
tinybvh's `suzanne`, `bunny` and `cryteksponza` scenes: node arrays and primitive indices bit for
bit, and 65,536 rays per scene through `Intersect`, `IsOccluded`, `Refit`, indexed meshes and a
TLAS. They also cover custom geometry and threaded builds.

```
dotnet test Tests -c Release
```

The scenes and dumps come from the tinybvh-unity repository: run its `TestData/fetch.ps1`, then
build and run `Tools/RefDump`. The tests look for them in `../tinybvh-unity/TestData`, or in the
directory named by `MICROBVH_TESTDATA`, and skip themselves when the data is missing. Add
`-p:MicroBvhCheckOverflow=true` to test the library built with overflow checking.

`MicroBVH.csproj` builds the file against its portability floor: .NET Standard 2.1, C# 9, unsafe
code disabled, warnings as errors.

## Deviations from tinybvh

- `BvhMath.Rcp` caps the reciprocal direction at 1e30 instead of `FLT_MAX`. The C++ relies on the
  slab test overflowing to NaN for axis-aligned rays, which only works when every intermediate is
  rounded to single precision.
- The partition step recomputes the bin index with the same vector expression as the binning
  pass, so both agree regardless of intermediate precision.
- Errors throw exceptions instead of calling `exit`, and array arguments are checked.
- The per-octant template variants of the traversal are one function with the ray sign
  predicates evaluated at runtime.
- A threaded build subdivides the top of the tree on the calling thread and builds the subtrees at
  depth 9 in one `Parallel.For`. The tree shape equals the serial one; the node numbering does not.
- A TLAS takes a `Bvh[]`, where tinybvh's `BVHBase**` can mix layouts; microBVH has only one.
- `Refit` on a BVH over custom geometry throws, since there are no vertices to refit to.
- Create instances with `BlasInstance.Create( blasIdx )`: a default `BlasInstance` has zero
  matrices, where the C++ defaults them to identity.
- `Intersection` leaves out tinybvh's `auxData` pointer and user-data union.
- Results are bit-exact where every float operation is rounded to single precision, as on .NET
  Core and .NET 5+. A runtime that evaluates float arithmetic at higher precision, such as Unity's
  Mono, can differ in the last bits.

## License

MIT, see `LICENSE.md`. tinybvh is (c) Jacco Bikker / Breda University of Applied Sciences.
