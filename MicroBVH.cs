/*
The MIT License (MIT)

Copyright (c) 2024-2026, Jacco Bikker / Breda University of Applied Sciences.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// microBVH: a small, portable C# subset of tinybvh 1.8.0, in a single file.
//
// How to use:
//
// Add this file to any project that targets .NET Standard 2.1 or later: .NET Core 3+ and .NET 5+,
// Mono, Unity 2021.2+ and Godot 4 all qualify. It is safe code only (no unsafe blocks, no pointers,
// no SIMD intrinsics), needs nothing beyond the base class library, and also compiles in projects
// that enable overflow checking.
//
// Build a BVH over a list of triangles and trace a ray:
//   BvhVec4[] vertices = ...;           // three vertices per triangle; w is ignored.
//   Bvh bvh = new Bvh();
//   bvh.Build( vertices, triangleCount );
//   Ray ray = new Ray( new BvhVec3( 0, 0, 0 ), new BvhVec3( 0, 0, 1 ) );
//   bvh.Intersect( ref ray );
// After this, intersection information is in ray.Hit; ray.Hit.T == BvhConstants.Far on a miss.
//
// What microBVH keeps of tinybvh:
// - binned SAH construction (tinybvh's reference builder, optionally on the thread pool) over
//   triangle soups, indexed meshes, custom primitives given as AABBs or through a callback, and
//   BLAS instances with 4x4 transforms (a TLAS);
// - closest-hit (Intersect) and any-hit (IsOccluded) traversal, with custom-geometry callbacks and
//   16-bit instance masks;
// - Refit for deforming meshes, and the SAH cost and node counts of a tree.
// What it leaves out: the SBVH and the other alternative builders, the optimizer, the wide and GPU
// layouts, SIMD traversal, double precision, voxel sets, opacity micromaps, ray packets, sphere
// queries and saving trees to disk. Use tinybvh itself for those.
//
// The geometry is referenced, not copied: keep the vertex array alive while the BVH is in use.
// Traversal is safe to run from many threads at once on the same BVH.
//
// microBVH produces the same trees and the same hits as tinybvh's scalar build, bit for bit, on
// runtimes that round every float operation to single precision (.NET Core and .NET 5+ do). Where
// it deviates from the C++, the code says so in a "Deviation:" note.
//
// tinybvh by Jacco Bikker and contributors: github.com/jbikker/tinybvh

using System.Globalization;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using System.Threading;
using System;

namespace MicroBVH
{
	// ============================================================================
	//
	//        C O N F I G U R A T I O N
	//
	// ============================================================================

	/// <summary>
	/// tinybvh's compile-time configuration (BVHBINS, C_INT, C_TRAV and so on). The C++ lets these
	/// be overridden with #defines before inclusion; here they are constants. The SAH costs are
	/// also fields of every BVH, which can be changed per instance.
	/// </summary>
	public static class BvhConstants
	{
		/// <summary>Miss distance, tinybvh's BVH_FAR.</summary>
		public const float Far = 1e30f;
		/// <summary>Bin count of the binned SAH builder (BVHBINS).</summary>
		public const int Bins = 8;
		/// <summary>Threaded builds hand the subtrees rooted at this depth to the thread pool (MT_SPAWN_DEPTH).</summary>
		public const int MtSpawnDepth = 9;
		/// <summary>Builds of fewer primitives than this stay single-threaded (MT_BUILD_THRESHOLD).</summary>
		public const uint MtBuildThreshold = 50000;
		/// <summary>Largest reciprocal-direction magnitude; see <see cref="BvhMath.Rcp(float)"/>.</summary>
		public const float RcpMax = 1e30f;
		/// <summary>Default ray mask: intersect every instance (RAY_MASK_INTERSECT_ALL).</summary>
		public const uint RayMaskIntersectAll = 0xFFFF;
		/// <summary>Default SAH cost of a traversal step (C_TRAV).</summary>
		public const float DefaultTraversalCost = 1f;
		/// <summary>Default SAH cost of a primitive intersection (C_INT).</summary>
		public const float DefaultIntersectionCost = 1f;
	}

	// ============================================================================
	//
	//        V E C T O R   T Y P E S
	//
	// ============================================================================

	/// <summary>
	/// Port of bvhvec3. The operators evaluate in the C++ operation order, so results match the
	/// original bit for bit on runtimes that round every float operation to single precision.
	/// </summary>
	public struct BvhVec3
	{
		public float x, y, z;

		public BvhVec3( float a )
		{
			x = a;
			y = a;
			z = a;
		}

		public BvhVec3( float a, float b, float c )
		{
			x = a;
			y = b;
			z = c;
		}

		/// <summary>Cell access: 0 = x, 1 = y, 2 = z.</summary>
		public float this[ int i ]
		{
			readonly get
			{
				return i == 0 ? x : ( i == 1 ? y : z );
			}
			set
			{
				if ( i == 0 )
				{
					x = value;
				}
				else if ( i == 1 )
				{
					y = value;
				}
				else
				{
					z = value;
				}
			}
		}

		/// <summary>The C++ bvhvec3( const bvhvec4 ) conversion, which drops w.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static implicit operator BvhVec3( BvhVec4 a )
		{
			return new BvhVec3( a.x, a.y, a.z );
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static BvhVec3 operator -( BvhVec3 a )
		{
			return new BvhVec3( -a.x, -a.y, -a.z );
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static BvhVec3 operator +( BvhVec3 a, BvhVec3 b )
		{
			return new BvhVec3( a.x + b.x, a.y + b.y, a.z + b.z );
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static BvhVec3 operator -( BvhVec3 a, BvhVec3 b )
		{
			return new BvhVec3( a.x - b.x, a.y - b.y, a.z - b.z );
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static BvhVec3 operator *( BvhVec3 a, BvhVec3 b )
		{
			return new BvhVec3( a.x * b.x, a.y * b.y, a.z * b.z );
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static BvhVec3 operator *( BvhVec3 a, float b )
		{
			return new BvhVec3( a.x * b, a.y * b, a.z * b );
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static BvhVec3 operator *( float b, BvhVec3 a )
		{
			return new BvhVec3( b * a.x, b * a.y, b * a.z );
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static BvhVec3 operator /( float b, BvhVec3 a )
		{
			return new BvhVec3( b / a.x, b / a.y, b / a.z );
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static BvhVec3 operator /( BvhVec3 b, BvhVec3 a )
		{
			return new BvhVec3( b.x / a.x, b.y / a.y, b.z / a.z );
		}

		public override readonly string ToString()
		{
			return string.Format( CultureInfo.InvariantCulture, "({0}, {1}, {2})", x, y, z );
		}
	}

	/// <summary>Port of bvhvec4. Vertices are passed as bvhvec4; w is ignored by every builder.</summary>
	public struct BvhVec4
	{
		public float x, y, z, w;

		public BvhVec4( float a )
		{
			x = a;
			y = a;
			z = a;
			w = a;
		}

		public BvhVec4( float a, float b, float c, float d )
		{
			x = a;
			y = b;
			z = c;
			w = d;
		}

		public BvhVec4( BvhVec3 a, float b )
		{
			x = a.x;
			y = a.y;
			z = a.z;
			w = b;
		}

		/// <summary>Cell access: 0 = x, 1 = y, 2 = z, 3 = w.</summary>
		public float this[ int i ]
		{
			readonly get
			{
				return i == 0 ? x : ( i == 1 ? y : ( i == 2 ? z : w ) );
			}
			set
			{
				if ( i == 0 )
				{
					x = value;
				}
				else if ( i == 1 )
				{
					y = value;
				}
				else if ( i == 2 )
				{
					z = value;
				}
				else
				{
					w = value;
				}
			}
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static BvhVec4 operator -( BvhVec4 a )
		{
			return new BvhVec4( -a.x, -a.y, -a.z, -a.w );
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static BvhVec4 operator +( BvhVec4 a, BvhVec4 b )
		{
			return new BvhVec4( a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w );
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static BvhVec4 operator -( BvhVec4 a, BvhVec4 b )
		{
			return new BvhVec4( a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w );
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static BvhVec4 operator *( BvhVec4 a, BvhVec4 b )
		{
			return new BvhVec4( a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w );
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static BvhVec4 operator *( BvhVec4 a, float b )
		{
			return new BvhVec4( a.x * b, a.y * b, a.z * b, a.w * b );
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static BvhVec4 operator *( float b, BvhVec4 a )
		{
			return new BvhVec4( b * a.x, b * a.y, b * a.z, b * a.w );
		}

		public override readonly string ToString()
		{
			return string.Format( CultureInfo.InvariantCulture, "({0}, {1}, {2}, {3})", x, y, z, w );
		}
	}

	/// <summary>Port of bvhint3.</summary>
	public struct BvhInt3
	{
		public int x, y, z;

		public BvhInt3( int a )
		{
			x = a;
			y = a;
			z = a;
		}

		public BvhInt3( int a, int b, int c )
		{
			x = a;
			y = b;
			z = c;
		}

		/// <summary>
		/// The C++ bvhint3( const bvhvec3&amp; ) conversion: truncates each component. Unchecked, so
		/// it also works in projects that enable overflow checking; every caller clamps the result.
		/// </summary>
		public BvhInt3( BvhVec3 a )
		{
			unchecked
			{
				x = ( int )a.x;
				y = ( int )a.y;
				z = ( int )a.z;
			}
		}

		/// <summary>Cell access: 0 = x, 1 = y, 2 = z.</summary>
		public int this[ int i ]
		{
			readonly get
			{
				return i == 0 ? x : ( i == 1 ? y : z );
			}
			set
			{
				if ( i == 0 )
				{
					x = value;
				}
				else if ( i == 1 )
				{
					y = value;
				}
				else
				{
					z = value;
				}
			}
		}

		public override readonly string ToString()
		{
			return string.Format( CultureInfo.InvariantCulture, "({0}, {1}, {2})", x, y, z );
		}
	}

	/// <summary>
	/// Port of bvhmat4: a row-major 4x4 matrix. Cell i is row i / 4, column i % 4, so the
	/// translation lives in cells 3, 7 and 11 (Row0.w, Row1.w and Row2.w).
	/// </summary>
	public struct BvhMat4
	{
		public BvhVec4 Row0;
		public BvhVec4 Row1;
		public BvhVec4 Row2;
		public BvhVec4 Row3;

		/// <summary>The identity matrix, the C++ default for bvhmat4.</summary>
		public static BvhMat4 Identity => new BvhMat4
		{
			Row0 = new BvhVec4( 1f, 0f, 0f, 0f ),
			Row1 = new BvhVec4( 0f, 1f, 0f, 0f ),
			Row2 = new BvhVec4( 0f, 0f, 1f, 0f ),
			Row3 = new BvhVec4( 0f, 0f, 0f, 1f )
		};

		/// <summary>Cell access in tinybvh order: index = row * 4 + column.</summary>
		public float this[ int i ]
		{
			readonly get
			{
				switch ( i >> 2 )
				{
					case 0: return Row0[ i & 3 ];
					case 1: return Row1[ i & 3 ];
					case 2: return Row2[ i & 3 ];
					default: return Row3[ i & 3 ];
				}
			}
			set
			{
				switch ( i >> 2 )
				{
					case 0: Row0[ i & 3 ] = value; break;
					case 1: Row1[ i & 3 ] = value; break;
					case 2: Row2[ i & 3 ] = value; break;
					default: Row3[ i & 3 ] = value; break;
				}
			}
		}

		/// <summary>Port of tinybvh_transform_point, same operation order.</summary>
		public readonly BvhVec3 TransformPoint( BvhVec3 v )
		{
			BvhVec3 res = new BvhVec3(
				( Row0.x * v.x ) + ( Row0.y * v.y ) + ( Row0.z * v.z ) + Row0.w,
				( Row1.x * v.x ) + ( Row1.y * v.y ) + ( Row1.z * v.z ) + Row1.w,
				( Row2.x * v.x ) + ( Row2.y * v.y ) + ( Row2.z * v.z ) + Row2.w );
			float w = ( Row3.x * v.x ) + ( Row3.y * v.y ) + ( Row3.z * v.z ) + Row3.w;
			if ( w == 1f )
			{
				return res;
			}
			return res * ( 1f / w );
		}

		/// <summary>Port of tinybvh_transform_vector, same operation order.</summary>
		public readonly BvhVec3 TransformVector( BvhVec3 v )
		{
			return new BvhVec3(
				( Row0.x * v.x ) + ( Row0.y * v.y ) + ( Row0.z * v.z ),
				( Row1.x * v.x ) + ( Row1.y * v.y ) + ( Row1.z * v.z ),
				( Row2.x * v.x ) + ( Row2.y * v.y ) + ( Row2.z * v.z ) );
		}
	}

	/// <summary>
	/// The tinybvh_* scalar and vector helpers. Min and Max are the C++ ternaries, not Math.Min /
	/// Math.Max, which treat NaN and signed zeros differently.
	/// </summary>
	public static class BvhMath
	{
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static float Min( float a, float b )
		{
			return a < b ? a : b;
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static float Max( float a, float b )
		{
			return a > b ? a : b;
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static BvhVec3 Min( BvhVec3 a, BvhVec3 b )
		{
			return new BvhVec3( Min( a.x, b.x ), Min( a.y, b.y ), Min( a.z, b.z ) );
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static BvhVec3 Max( BvhVec3 a, BvhVec3 b )
		{
			return new BvhVec3( Max( a.x, b.x ), Max( a.y, b.y ), Max( a.z, b.z ) );
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static BvhVec4 Min( BvhVec4 a, BvhVec4 b )
		{
			return new BvhVec4( Min( a.x, b.x ), Min( a.y, b.y ), Min( a.z, b.z ), Min( a.w, b.w ) );
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static BvhVec4 Max( BvhVec4 a, BvhVec4 b )
		{
			return new BvhVec4( Max( a.x, b.x ), Max( a.y, b.y ), Max( a.z, b.z ), Max( a.w, b.w ) );
		}

		/// <summary>Port of tinybvh_clamp for integers.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static int Clamp( int x, int a, int b )
		{
			return x > a ? ( x < b ? x : b ) : a;
		}

		/// <summary>Port of tinybvh_swap.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static void Swap<T>( ref T a, ref T b )
		{
			T t = a;
			a = b;
			b = t;
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static float Dot( BvhVec3 a, BvhVec3 b )
		{
			return ( a.x * b.x ) + ( a.y * b.y ) + ( a.z * b.z );
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static BvhVec3 Cross( BvhVec3 a, BvhVec3 b )
		{
			return new BvhVec3( ( a.y * b.z ) - ( a.z * b.y ), ( a.z * b.x ) - ( a.x * b.z ), ( a.x * b.y ) - ( a.y * b.x ) );
		}

		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static float Length( BvhVec3 a )
		{
			return MathF.Sqrt( ( a.x * a.x ) + ( a.y * a.y ) + ( a.z * a.z ) );
		}

		/// <summary>Port of tinybvh_normalize: a zero vector stays zero.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static BvhVec3 Normalize( BvhVec3 a )
		{
			float l = Length( a ), rl = l == 0f ? 0f : ( 1f / l );
			return a * rl;
		}

		/// <summary>Port of tinybvh_halfarea, including the guard for empty (inverted) bins.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static float HalfArea( BvhVec3 v )
		{
			return v.x < -BvhConstants.Far ? 0f : ( ( v.x * v.y ) + ( v.y * v.z ) + ( v.z * v.x ) );
		}

		/// <summary>
		/// Port of tinybvh_safercp: 1/x, or a huge value with the sign of x when that is not finite.
		/// Deviation: the C++ returns +-FLT_MAX, which relies on the slab test's origin term overflowing
		/// to infinity and the resulting NaNs falling through min/max. That only works when every
		/// intermediate is rounded to single precision, which not every .NET runtime does, so the
		/// magnitude is capped at RcpMax instead: large enough to cull any box the ray is outside of
		/// on that axis, small enough that products with scene coordinates stay finite.
		/// </summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static float Rcp( float x )
		{
			float r = 1f / x;
			if ( !( MathF.Abs( r ) < BvhConstants.RcpMax ) )
			{
				r = ( x < 0f || ( x == 0f && ( AsUint( x ) & 0x80000000u ) != 0 ) ) ? -BvhConstants.RcpMax : BvhConstants.RcpMax;
			}
			return r;
		}

		/// <summary>Port of tinybvh_rcp for bvhvec3; see <see cref="Rcp(float)"/>.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static BvhVec3 Rcp( BvhVec3 a )
		{
			return new BvhVec3( Rcp( a.x ), Rcp( a.y ), Rcp( a.z ) );
		}

		/// <summary>Port of BVHBase::SA, same operation order.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static float SurfaceArea( BvhVec3 aabbMin, BvhVec3 aabbMax )
		{
			BvhVec3 e = aabbMax - aabbMin;
			return ( e.x * e.y ) + ( e.y * e.z ) + ( e.z * e.x );
		}

		/// <summary>
		/// Port of tinybvh_intersect_aabb (slab test): the entry distance of the ray into the box, or
		/// Far on a miss. Handy in custom-geometry callbacks for primitives given as boxes.
		/// </summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static float IntersectAabb( in Ray ray, BvhVec3 aabbMin, BvhVec3 aabbMax )
		{
			float tx1 = ( aabbMin.x - ray.O.x ) * ray.RD.x, tx2 = ( aabbMax.x - ray.O.x ) * ray.RD.x;
			float tmin = Min( tx1, tx2 ), tmax = Max( tx1, tx2 );
			float ty1 = ( aabbMin.y - ray.O.y ) * ray.RD.y, ty2 = ( aabbMax.y - ray.O.y ) * ray.RD.y;
			tmin = Max( tmin, Min( ty1, ty2 ) );
			tmax = Min( tmax, Max( ty1, ty2 ) );
			float tz1 = ( aabbMin.z - ray.O.z ) * ray.RD.z, tz2 = ( aabbMax.z - ray.O.z ) * ray.RD.z;
			tmin = Max( tmin, Min( tz1, tz2 ) );
			tmax = Min( tmax, Max( tz1, tz2 ) );
			if ( tmax >= tmin && tmin < ray.Hit.T && tmax >= 0f )
			{
				return tmin;
			}
			return BvhConstants.Far;
		}

		/// <summary>The bits of a float, as a uint.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static uint AsUint( float f )
		{
			return unchecked( ( uint )BitConverter.SingleToInt32Bits( f ) );
		}

		/// <summary>The float with the given bits.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public static float AsFloat( uint u )
		{
			return BitConverter.Int32BitsToSingle( unchecked( ( int )u ) );
		}
	}

	// ============================================================================
	//
	//        R A Y S ,   N O D E S   A N D   I N S T A N C E S
	//
	// ============================================================================

	/// <summary>
	/// Intersection record. The C++ struct also carries an auxData pointer and 56 bytes of user
	/// data; those are left out here.
	/// </summary>
	public struct Intersection
	{
		/// <summary>Instance index, set by TLAS traversal (INST_IDX_BITS == 32: stored in its own field).</summary>
		public uint Inst;
		/// <summary>Distance along the ray; BvhConstants.Far when nothing was hit.</summary>
		public float T;
		/// <summary>Barycentric coordinates of the hit.</summary>
		public float U, V;
		/// <summary>Primitive index.</summary>
		public uint Prim;
	}

	/// <summary>
	/// Ray with precomputed reciprocal direction. For a single BLAS, RD must be kept in sync with D;
	/// TLAS traversal recomputes it for every BLAS it enters.
	/// </summary>
	public struct Ray
	{
		public BvhVec3 O;
		/// <summary>16-bit instance mask; compared against BlasInstance.Mask during TLAS traversal.</summary>
		public uint Mask;
		public BvhVec3 D;
		public uint InstIdx;
		/// <summary>Reciprocal direction, see <see cref="BvhMath.Rcp(BvhVec3)"/>.</summary>
		public BvhVec3 RD;
		public Intersection Hit;

		/// <summary>Port of the Ray constructor: normalizes the direction and sets up its reciprocal.</summary>
		public Ray( BvhVec3 origin, BvhVec3 direction, float t = BvhConstants.Far, uint mask = BvhConstants.RayMaskIntersectAll )
		{
			O = origin;
			D = BvhMath.Normalize( direction );
			RD = BvhMath.Rcp( D );
			Mask = mask & BvhConstants.RayMaskIntersectAll;
			InstIdx = 0;
			Hit = default;
			Hit.T = t;
		}
	}

	/// <summary>Wald 32-byte BVH node (BVH::BVHNode). Two fit in a cache line.</summary>
	public struct BvhNode
	{
		public BvhVec3 AabbMin;
		/// <summary>Index of the left child for interior nodes, of the first primitive index for leaves.</summary>
		public uint LeftFirst;
		public BvhVec3 AabbMax;
		/// <summary>Primitive count; zero for interior nodes.</summary>
		public uint TriCount;

		public readonly bool IsLeaf => TriCount > 0;
		public readonly float SurfaceArea => BvhMath.SurfaceArea( AabbMin, AabbMax );
	}

	/// <summary>
	/// Bounds of an input primitive, as used during construction (BVHBase::Fragment). The C++ adds a
	/// 'clipped' flag for spatial splits, which this library does not do.
	/// </summary>
	public struct Fragment
	{
		public BvhVec3 BMin;
		/// <summary>Index of the original primitive.</summary>
		public uint PrimIdx;
		public BvhVec3 BMax;
	}

	/// <summary>
	/// A BLAS placed in a TLAS with a transform (BLASInstance). The TLAS is built over the
	/// world-space bounds of the instances. Create instances with <see cref="Create"/>: a default
	/// instance has all-zero matrices, where the C++ defaults them to identity.
	/// </summary>
	public struct BlasInstance
	{
		public BvhMat4 Transform;
		public BvhMat4 InvTransform;
		/// <summary>World-space bounds, computed by Update.</summary>
		public BvhVec3 AabbMin;
		public uint BlasIdx;
		public BvhVec3 AabbMax;
		public uint Mask;

		/// <summary>Port of BLASInstance( uint32_t idx ), including the C++ member initializers.</summary>
		public static BlasInstance Create( uint blasIdx )
		{
			return new BlasInstance
			{
				Transform = BvhMat4.Identity,
				InvTransform = BvhMat4.Identity,
				AabbMin = new BvhVec3( BvhConstants.Far ),
				BlasIdx = blasIdx,
				AabbMax = new BvhVec3( -BvhConstants.Far ),
				Mask = BvhConstants.RayMaskIntersectAll
			};
		}

		/// <summary>
		/// Port of BLASInstance::Update: inverts the transform and computes the world-space bounds
		/// of the BLAS root bounds under it. Only the root bounds of the BLAS are read.
		/// </summary>
		public void Update( Bvh blas )
		{
			InvertTransform(); // TODO: done unconditionally; for a big TLAS this may be wasteful.
			// transform the eight corners of the root node aabb using the
			// instance transform and calculate the worldspace aabb over those.
			AabbMin = new BvhVec3( BvhConstants.Far );
			AabbMax = new BvhVec3( -BvhConstants.Far );
			BvhVec3 bmin = blas.AabbMin, bmax = blas.AabbMax;
			for ( int j = 0; j < 8; j++ )
			{
				BvhVec3 p = new BvhVec3(
					( j & 1 ) != 0 ? bmax.x : bmin.x,
					( j & 2 ) != 0 ? bmax.y : bmin.y,
					( j & 4 ) != 0 ? bmax.z : bmin.z );
				BvhVec3 t = Transform.TransformPoint( p );
				AabbMin = BvhMath.Min( AabbMin, t );
				AabbMax = BvhMath.Max( AabbMax, t );
			}
		}

		/// <summary>Port of BLASInstance::InvertTransform: stores the inverse of Transform in InvTransform.</summary>
		public void InvertTransform()
		{
			// math from MESA, via http://stackoverflow.com/questions/1148309/inverting-a-4x4-matrix
			Span<float> T = stackalloc float[ 16 ];
			Span<float> iT = stackalloc float[ 16 ];
			for ( int i = 0; i < 16; i++ )
			{
				T[ i ] = Transform[ i ];
			}
			iT[ 0 ] = ( T[ 5 ] * T[ 10 ] * T[ 15 ] ) - ( T[ 5 ] * T[ 11 ] * T[ 14 ] ) - ( T[ 9 ] * T[ 6 ] * T[ 15 ] ) + ( T[ 9 ] * T[ 7 ] * T[ 14 ] ) + ( T[ 13 ] * T[ 6 ] * T[ 11 ] ) - ( T[ 13 ] * T[ 7 ] * T[ 10 ] );
			iT[ 1 ] = -( T[ 1 ] * T[ 10 ] * T[ 15 ] ) + ( T[ 1 ] * T[ 11 ] * T[ 14 ] ) + ( T[ 9 ] * T[ 2 ] * T[ 15 ] ) - ( T[ 9 ] * T[ 3 ] * T[ 14 ] ) - ( T[ 13 ] * T[ 2 ] * T[ 11 ] ) + ( T[ 13 ] * T[ 3 ] * T[ 10 ] );
			iT[ 2 ] = ( T[ 1 ] * T[ 6 ] * T[ 15 ] ) - ( T[ 1 ] * T[ 7 ] * T[ 14 ] ) - ( T[ 5 ] * T[ 2 ] * T[ 15 ] ) + ( T[ 5 ] * T[ 3 ] * T[ 14 ] ) + ( T[ 13 ] * T[ 2 ] * T[ 7 ] ) - ( T[ 13 ] * T[ 3 ] * T[ 6 ] );
			iT[ 3 ] = -( T[ 1 ] * T[ 6 ] * T[ 11 ] ) + ( T[ 1 ] * T[ 7 ] * T[ 10 ] ) + ( T[ 5 ] * T[ 2 ] * T[ 11 ] ) - ( T[ 5 ] * T[ 3 ] * T[ 10 ] ) - ( T[ 9 ] * T[ 2 ] * T[ 7 ] ) + ( T[ 9 ] * T[ 3 ] * T[ 6 ] );
			iT[ 4 ] = -( T[ 4 ] * T[ 10 ] * T[ 15 ] ) + ( T[ 4 ] * T[ 11 ] * T[ 14 ] ) + ( T[ 8 ] * T[ 6 ] * T[ 15 ] ) - ( T[ 8 ] * T[ 7 ] * T[ 14 ] ) - ( T[ 12 ] * T[ 6 ] * T[ 11 ] ) + ( T[ 12 ] * T[ 7 ] * T[ 10 ] );
			iT[ 5 ] = ( T[ 0 ] * T[ 10 ] * T[ 15 ] ) - ( T[ 0 ] * T[ 11 ] * T[ 14 ] ) - ( T[ 8 ] * T[ 2 ] * T[ 15 ] ) + ( T[ 8 ] * T[ 3 ] * T[ 14 ] ) + ( T[ 12 ] * T[ 2 ] * T[ 11 ] ) - ( T[ 12 ] * T[ 3 ] * T[ 10 ] );
			iT[ 6 ] = -( T[ 0 ] * T[ 6 ] * T[ 15 ] ) + ( T[ 0 ] * T[ 7 ] * T[ 14 ] ) + ( T[ 4 ] * T[ 2 ] * T[ 15 ] ) - ( T[ 4 ] * T[ 3 ] * T[ 14 ] ) - ( T[ 12 ] * T[ 2 ] * T[ 7 ] ) + ( T[ 12 ] * T[ 3 ] * T[ 6 ] );
			iT[ 7 ] = ( T[ 0 ] * T[ 6 ] * T[ 11 ] ) - ( T[ 0 ] * T[ 7 ] * T[ 10 ] ) - ( T[ 4 ] * T[ 2 ] * T[ 11 ] ) + ( T[ 4 ] * T[ 3 ] * T[ 10 ] ) + ( T[ 8 ] * T[ 2 ] * T[ 7 ] ) - ( T[ 8 ] * T[ 3 ] * T[ 6 ] );
			iT[ 8 ] = ( T[ 4 ] * T[ 9 ] * T[ 15 ] ) - ( T[ 4 ] * T[ 11 ] * T[ 13 ] ) - ( T[ 8 ] * T[ 5 ] * T[ 15 ] ) + ( T[ 8 ] * T[ 7 ] * T[ 13 ] ) + ( T[ 12 ] * T[ 5 ] * T[ 11 ] ) - ( T[ 12 ] * T[ 7 ] * T[ 9 ] );
			iT[ 9 ] = -( T[ 0 ] * T[ 9 ] * T[ 15 ] ) + ( T[ 0 ] * T[ 11 ] * T[ 13 ] ) + ( T[ 8 ] * T[ 1 ] * T[ 15 ] ) - ( T[ 8 ] * T[ 3 ] * T[ 13 ] ) - ( T[ 12 ] * T[ 1 ] * T[ 11 ] ) + ( T[ 12 ] * T[ 3 ] * T[ 9 ] );
			iT[ 10 ] = ( T[ 0 ] * T[ 5 ] * T[ 15 ] ) - ( T[ 0 ] * T[ 7 ] * T[ 13 ] ) - ( T[ 4 ] * T[ 1 ] * T[ 15 ] ) + ( T[ 4 ] * T[ 3 ] * T[ 13 ] ) + ( T[ 12 ] * T[ 1 ] * T[ 7 ] ) - ( T[ 12 ] * T[ 3 ] * T[ 5 ] );
			iT[ 11 ] = -( T[ 0 ] * T[ 5 ] * T[ 11 ] ) + ( T[ 0 ] * T[ 7 ] * T[ 9 ] ) + ( T[ 4 ] * T[ 1 ] * T[ 11 ] ) - ( T[ 4 ] * T[ 3 ] * T[ 9 ] ) - ( T[ 8 ] * T[ 1 ] * T[ 7 ] ) + ( T[ 8 ] * T[ 3 ] * T[ 5 ] );
			iT[ 12 ] = -( T[ 4 ] * T[ 9 ] * T[ 14 ] ) + ( T[ 4 ] * T[ 10 ] * T[ 13 ] ) + ( T[ 8 ] * T[ 5 ] * T[ 14 ] ) - ( T[ 8 ] * T[ 6 ] * T[ 13 ] ) - ( T[ 12 ] * T[ 5 ] * T[ 10 ] ) + ( T[ 12 ] * T[ 6 ] * T[ 9 ] );
			iT[ 13 ] = ( T[ 0 ] * T[ 9 ] * T[ 14 ] ) - ( T[ 0 ] * T[ 10 ] * T[ 13 ] ) - ( T[ 8 ] * T[ 1 ] * T[ 14 ] ) + ( T[ 8 ] * T[ 2 ] * T[ 13 ] ) + ( T[ 12 ] * T[ 1 ] * T[ 10 ] ) - ( T[ 12 ] * T[ 2 ] * T[ 9 ] );
			iT[ 14 ] = -( T[ 0 ] * T[ 5 ] * T[ 14 ] ) + ( T[ 0 ] * T[ 6 ] * T[ 13 ] ) + ( T[ 4 ] * T[ 1 ] * T[ 14 ] ) - ( T[ 4 ] * T[ 2 ] * T[ 13 ] ) - ( T[ 12 ] * T[ 1 ] * T[ 6 ] ) + ( T[ 12 ] * T[ 2 ] * T[ 5 ] );
			iT[ 15 ] = ( T[ 0 ] * T[ 5 ] * T[ 10 ] ) - ( T[ 0 ] * T[ 6 ] * T[ 9 ] ) - ( T[ 4 ] * T[ 1 ] * T[ 10 ] ) + ( T[ 4 ] * T[ 2 ] * T[ 9 ] ) + ( T[ 8 ] * T[ 1 ] * T[ 6 ] ) - ( T[ 8 ] * T[ 2 ] * T[ 5 ] );
			// the C++ writes the cofactors straight into invTransform, so they stay there on failure.
			for ( int i = 0; i < 16; i++ )
			{
				InvTransform[ i ] = iT[ i ];
			}
			float det = ( T[ 0 ] * iT[ 0 ] ) + ( T[ 1 ] * iT[ 4 ] ) + ( T[ 2 ] * iT[ 8 ] ) + ( T[ 3 ] * iT[ 12 ] );
			if ( det == 0f )
			{
				return; // actually, invert failed. That's bad.
			}
			float invdet = 1f / det;
			for ( int i = 0; i < 16; i++ )
			{
				InvTransform[ i ] = iT[ i ] * invdet;
			}
		}
	}

	// ============================================================================
	//
	//        C U S T O M   G E O M E T R Y
	//
	// ============================================================================

	/// <summary>Fills the bounds of one custom primitive; the customGetAABB callback of BVH::Build.</summary>
	public delegate void GetAabbDelegate( uint prim, out BvhVec3 aabbMin, out BvhVec3 aabbMax );

	/// <summary>
	/// tinybvh's customIntersect callback: intersects one custom primitive and updates ray.Hit when
	/// the hit is closer than ray.Hit.T. Returns true when it registered a hit.
	/// </summary>
	public delegate bool CustomIntersectDelegate( ref Ray ray, uint prim );

	/// <summary>tinybvh's customIsOccluded callback: true when the primitive blocks the ray within ray.Hit.T.</summary>
	public delegate bool CustomOccludedDelegate( in Ray ray, uint prim );

	// ============================================================================
	//
	//        B V H
	//
	// ============================================================================

	/// <summary>
	/// Port of tinybvh's BVH class: a binary BVH in the Wald 32-byte node layout, over triangles,
	/// custom primitives or BLAS instances (a TLAS). This section holds the data; construction and
	/// traversal live in the sections below. Input geometry is referenced, not owned: keep the
	/// vertex array alive, and unchanged unless refitting, while the BVH is in use.
	/// </summary>
	public sealed partial class Bvh
	{
		/// <summary>Cost of a traversal step, used to steer SAH construction (c_trav).</summary>
		public float TraversalCost = BvhConstants.DefaultTraversalCost;
		/// <summary>Cost of a primitive intersection, used to steer SAH construction (c_int).</summary>
		public float IntersectionCost = BvhConstants.DefaultIntersectionCost;

		/// <summary>Number of primitives (triangles, custom primitives or instances) in the BVH.</summary>
		public uint TriCount;
		/// <summary>Bounds of the root node of the BVH.</summary>
		public BvhVec3 AabbMin, AabbMax;

		/// <summary>Input vertices (not owned): three per triangle, or addressed through VertIdx. Only xyz is used.</summary>
		public BvhVec4[] Verts;
		/// <summary>Vertex indices (not owned), three per primitive; null for a triangle soup.</summary>
		public uint[] VertIdx;

		/// <summary>Node pool. Root is always node 0; node 1 is unused, for cache line alignment.</summary>
		public BvhNode[] Nodes;
		public uint UsedNodes;
		/// <summary>Primitive index array: leaves reference the ranges of this.</summary>
		public uint[] PrimIdx;
		/// <summary>Input primitive bounds.</summary>
		public Fragment[] Fragments;

		/// <summary>
		/// Subdivide on the thread pool (tinybvh's threaded build). Builds of fewer than
		/// BvhConstants.MtBuildThreshold primitives stay serial. Node pairs are handed out by an
		/// atomic counter, so the node numbering of a threaded build is not reproducible; the tree
		/// shape is.
		/// </summary>
		public bool UseThreadedBuild;
		/// <summary>Number of subtrees the last build handed to the thread pool; zero when it ran serially.</summary>
		public uint ThreadedSubtrees;

		/// <summary>
		/// Custom geometry callbacks (customIntersect / customIsOccluded). When set, traversal hands
		/// every primitive index in a leaf to these instead of intersecting a triangle.
		/// </summary>
		public CustomIntersectDelegate CustomIntersect;
		public CustomOccludedDelegate CustomIsOccluded;

		// TLAS data (not owned): set when this BVH was built over BLAS instances.
		public BlasInstance[] Instances;
		public uint InstanceCount;
		/// <summary>The BLASses the instances refer to.</summary>
		public Bvh[] Blasses;
		public uint BlasCount;

		/// <summary>Capacity of the node pool.</summary>
		public uint AllocatedNodes => Nodes == null ? 0 : ( uint )Nodes.Length;

		public bool IsTlas => Instances != null;

		/// <summary>Port of GET_PRIM_INDICES_I0_I1_I2: vertex indices of a primitive, indexed or not.</summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		public void GetPrimIndices( uint prim, out uint i0, out uint i1, out uint i2 )
		{
			if ( VertIdx != null )
			{
				i0 = VertIdx[ prim * 3 ];
				i1 = VertIdx[ ( prim * 3 ) + 1 ];
				i2 = VertIdx[ ( prim * 3 ) + 2 ];
			}
			else
			{
				i0 = prim * 3;
				i1 = ( prim * 3 ) + 1;
				i2 = ( prim * 3 ) + 2;
			}
		}

		/// <summary>Ensures the node pool can hold count nodes. Contents are not preserved when it grows.</summary>
		private void AllocateNodes( uint count )
		{
			if ( AllocatedNodes < count )
			{
				Nodes = new BvhNode[ count ];
			}
		}

		/// <summary>Ensures the primitive index array can hold count entries. Contents are not preserved when it grows.</summary>
		private void AllocatePrimIdx( uint count )
		{
			if ( PrimIdx == null || PrimIdx.Length < count )
			{
				PrimIdx = new uint[ count ];
			}
		}

		/// <summary>Ensures the fragment array can hold count entries. Contents are not preserved when it grows.</summary>
		private void AllocateFragments( uint count )
		{
			if ( Fragments == null || Fragments.Length < count )
			{
				Fragments = new Fragment[ count ];
			}
		}
	}

	// ============================================================================
	//
	//        B V H   C O N S T R U C T I O N
	//
	// ============================================================================

	/// <summary>
	/// Construction half of tinybvh's BVH class: the Build entry points, PrepareBuild plus the
	/// binned SAH reference builder, the TLAS builder, Refit and the tree statistics. The threaded
	/// variant of the subdivision loop lives in the threaded construction section.
	/// </summary>
	public sealed partial class Bvh
	{
		/// <summary>
		/// The C++ newNodePtr member: the node pair allocator of the binned builder. Threaded builds
		/// draw from it concurrently, so it is only ever advanced with Interlocked.Add.
		/// </summary>
		private int newNodePtr;

		// BVH builder for triangle geometry.
		// This code uses no SIMD instructions; it is tinybvh's scalar reference builder.

		/// <summary>
		/// Builds over a triangle soup: three consecutive vertices per primitive. The vertex array
		/// may hold more than primCount * 3 vertices; the rest is ignored.
		/// </summary>
		public void Build( BvhVec4[] vertices, uint primCount )
		{
			ValidateBuildInput( vertices, null, primCount );
			PrepareBuild( vertices, null, primCount );
			RunBinnedBuild();
		}

		/// <summary>Builds over indexed triangles: three vertex indices per primitive.</summary>
		public void Build( BvhVec4[] vertices, uint[] indices, uint primCount )
		{
			if ( indices == null )
			{
				throw new ArgumentException( "Bvh.Build( .. ), indices == null.", nameof( indices ) );
			}
			ValidateBuildInput( vertices, indices, primCount );
			PrepareBuild( vertices, indices, primCount );
			RunBinnedBuild();
		}

		/// <summary>
		/// Runs the binned builder over prepared fragments, threaded when requested and the input is
		/// large enough; also used by the TLAS and custom-geometry builds.
		/// </summary>
		private void RunBinnedBuild()
		{
			if ( UseThreadedBuild && TriCount >= BvhConstants.MtBuildThreshold )
			{
				BuildBinnedThreaded();
			}
			else
			{
				BuildBinned();
			}
		}

		/// <summary>
		/// TLAS builder: builds a BVH over a list of BLAS instances, the C++
		/// BVH::Build( BLASInstance*, uint32_t, BVHBase**, uint32_t ). Every instance is updated
		/// against the BLAS it refers to (inverse transform and world-space bounds), unless blasses is
		/// null: then the instances are assumed to have been updated elsewhere, as in the C++. The
		/// TLAS references both arrays, it does not copy them. Like any other binned build, a TLAS of
		/// BvhConstants.MtBuildThreshold instances or more is built on the thread pool when
		/// UseThreadedBuild is set.
		/// </summary>
		public void BuildTlas( BlasInstance[] instances, uint instanceCount, Bvh[] blasses, uint blasCount )
		{
			if ( instances == null )
			{
				throw new ArgumentException( "Bvh.BuildTlas( .. ), instances == null.", nameof( instances ) );
			}
			if ( instanceCount == 0 )
			{
				throw new ArgumentException( "Bvh.BuildTlas( .. ), instanceCount == 0.", nameof( instanceCount ) );
			}
			if ( instances.Length < instanceCount )
			{
				throw new ArgumentException( "Bvh.BuildTlas( .. ), instances holds fewer than instanceCount entries.", nameof( instances ) );
			}
			if ( blasses != null && blasses.Length < blasCount )
			{
				throw new ArgumentException( "Bvh.BuildTlas( .. ), blasses holds fewer than blasCount entries.", nameof( blasses ) );
			}
			PrepareTlasBuild( instances, instanceCount, blasses, blasCount );
			RunBinnedBuild();
		}

		// Refitting: For animated meshes, where the topology remains intact. This
		// includes trees waving in the wind, or subsequent frames for skinned
		// animations. Repeated refitting tends to lead to deteriorated BVHs and
		// slower ray tracing. Rebuild when this happens.

		/// <summary>
		/// Port of BVH::Refit: recomputes every node's bounds from the current vertex positions,
		/// bottom-up. nodeIdx is unused, as in the C++. A BVH over custom geometry has no vertices
		/// to refit to: rebuild it instead.
		/// </summary>
		public void Refit( uint nodeIdx = 0 )
		{
			if ( Nodes == null )
			{
				throw new InvalidOperationException( "Bvh.Refit( .. ), nodes == null." );
			}
			if ( IsTlas )
			{
				throw new InvalidOperationException( "Bvh.Refit( .. ), do not refit a TLAS, use BuildTlas( .. )." );
			}
			if ( Verts == null )
			{
				throw new InvalidOperationException( "Bvh.Refit( .. ), a bvh over custom geometry has no vertices; rebuild it instead." );
			}
			BvhNode[] bvhNode = Nodes;
			uint[] primIdx = PrimIdx;
			uint[] vertIdx = VertIdx;
			BvhVec4[] verts = Verts;
			for ( int i = ( int )UsedNodes - 1; i >= 0; i-- )
			{
				if ( i != 1 )
				{
					ref BvhNode node = ref bvhNode[ i ];
					if ( node.IsLeaf ) // leaf: adjust to current triangle vertex positions
					{
						BvhVec4 bmin = new BvhVec4( BvhConstants.Far ), bmax = new BvhVec4( -BvhConstants.Far );
						if ( vertIdx != null )
						{
							for ( uint first = node.LeftFirst, j = 0; j < node.TriCount; j++ )
							{
								uint vidx = primIdx[ first + j ] * 3;
								uint i0 = vertIdx[ vidx ], i1 = vertIdx[ vidx + 1 ], i2 = vertIdx[ vidx + 2 ];
								BvhVec4 v0 = verts[ i0 ], v1 = verts[ i1 ], v2 = verts[ i2 ];
								BvhVec4 t1 = BvhMath.Min( v0, bmin ), t2 = BvhMath.Max( v0, bmax );
								BvhVec4 t3 = BvhMath.Min( v1, v2 ), t4 = BvhMath.Max( v1, v2 );
								bmin = BvhMath.Min( t1, t3 );
								bmax = BvhMath.Max( t2, t4 );
							}
						}
						else
						{
							for ( uint first = node.LeftFirst, j = 0; j < node.TriCount; j++ )
							{
								uint vidx = primIdx[ first + j ] * 3;
								BvhVec4 v0 = verts[ vidx ], v1 = verts[ vidx + 1 ], v2 = verts[ vidx + 2 ];
								BvhVec4 t1 = BvhMath.Min( v0, bmin ), t2 = BvhMath.Max( v0, bmax );
								BvhVec4 t3 = BvhMath.Min( v1, v2 ), t4 = BvhMath.Max( v1, v2 );
								bmin = BvhMath.Min( t1, t3 );
								bmax = BvhMath.Max( t2, t4 );
							}
						}
						node.AabbMin = bmin;
						node.AabbMax = bmax;
						continue;
					}
					// interior node: adjust to child bounds
					ref BvhNode left = ref bvhNode[ node.LeftFirst ];
					ref BvhNode right = ref bvhNode[ node.LeftFirst + 1 ];
					node.AabbMin = BvhMath.Min( left.AabbMin, right.AabbMin );
					node.AabbMax = BvhMath.Max( left.AabbMax, right.AabbMax );
				}
			}
			AabbMin = bvhNode[ 0 ].AabbMin;
			AabbMax = bvhNode[ 0 ].AabbMax;
		}

		/// <summary>
		/// Port of BVH::SAHCost: determine the SAH cost of the tree. This provides an indication
		/// of the quality of the BVH: Lower is better. Same recursion and summation order as the
		/// single-threaded C++ variant.
		/// </summary>
		public float SahCost( uint nodeIdx = 0 )
		{
			ref BvhNode n = ref Nodes[ nodeIdx ];
			if ( n.IsLeaf )
			{
				return IntersectionCost * n.SurfaceArea * n.TriCount;
			}
			float cost = ( TraversalCost * n.SurfaceArea ) + SahCost( n.LeftFirst ) + SahCost( n.LeftFirst + 1 );
			return nodeIdx == 0 ? ( cost / n.SurfaceArea ) : cost;
		}

		/// <summary>
		/// Determine the number of nodes in the tree. Typically the result should
		/// be usedNodes - 1 (second node is always unused), but some builders may
		/// have unused nodes besides node 1.
		/// </summary>
		public int NodeCount()
		{
			uint retVal = 0, nodeIdx = 0;
			int stackPtr = 0;
			Span<uint> stack = stackalloc uint[ 64 ];
			while ( true )
			{
				ref BvhNode n = ref Nodes[ nodeIdx ];
				retVal++;
				if ( n.IsLeaf )
				{
					if ( stackPtr == 0 )
					{
						break;
					}
					nodeIdx = stack[ --stackPtr ];
				}
				else
				{
					nodeIdx = n.LeftFirst;
					stack[ stackPtr++ ] = n.LeftFirst + 1;
				}
			}
			return ( int )retVal;
		}

		/// <summary>Determine the total number of primitives / fragments in leaf nodes.</summary>
		public int PrimCount( uint nodeIdx = 0 )
		{
			ref BvhNode n = ref Nodes[ nodeIdx ];
			return n.IsLeaf ? ( int )n.TriCount : ( PrimCount( n.LeftFirst ) + PrimCount( n.LeftFirst + 1 ) );
		}

		/// <summary>Determine the number of leaf nodes in the tree.</summary>
		public int LeafCount( uint nodeIdx = 0 )
		{
			uint retVal = 0;
			int stackPtr = 0;
			Span<uint> stack = stackalloc uint[ 64 ];
			while ( true )
			{
				ref BvhNode n = ref Nodes[ nodeIdx ];
				if ( n.IsLeaf )
				{
					retVal++;
					if ( stackPtr == 0 )
					{
						break;
					}
					nodeIdx = stack[ --stackPtr ];
				}
				else
				{
					nodeIdx = n.LeftFirst;
					stack[ stackPtr++ ] = n.LeftFirst + 1;
				}
			}
			return ( int )retVal;
		}

		/// <summary>Managed-side replacement for the BVH_FATAL_ERROR checks in BVH::PrepareBuild.</summary>
		private void ValidateBuildInput( BvhVec4[] vertices, uint[] indices, uint primCount )
		{
			if ( vertices == null )
			{
				throw new ArgumentException( "Bvh.Build( .. ), vertices == null.", nameof( vertices ) );
			}
			if ( vertices.Length == 0 )
			{
				throw new ArgumentException( "Bvh.Build( .. ), empty vertex slice.", nameof( vertices ) );
			}
			if ( primCount == 0 )
			{
				throw new ArgumentException( "Bvh.Build( .. ), primCount == 0.", nameof( primCount ) );
			}
			if ( indices == null && vertices.Length < ( long )primCount * 3 )
			{
				throw new ArgumentException( "Bvh.Build( .. ), vertices holds fewer than three vertices per primitive.", nameof( vertices ) );
			}
			if ( indices != null && indices.Length < ( long )primCount * 3 )
			{
				throw new ArgumentException( "Bvh.Build( .. ), indices holds fewer than three indices per primitive.", nameof( indices ) );
			}
		}

		/// <summary>
		/// Port of BVH::PrepareBuild: allocate memory and prepare a list of fragments to build a BVH
		/// over. indices is null for a triangle soup. Leaves UsedNodes at 2, the first free node,
		/// which is where the builders start allocating.
		/// </summary>
		private void PrepareBuild( BvhVec4[] vertices, uint[] indices, uint primCount )
		{
			uint spaceNeeded = primCount * 2; // upper limit
			// allocate memory on first build
			AllocateNodes( spaceNeeded );
			AllocatePrimIdx( primCount );
			AllocateFragments( primCount );
			Nodes[ 1 ] = default; // node 1 remains unused, for cache line alignment.
			// set verts, vertIdx
			TriCount = primCount;
			Verts = vertices;
			VertIdx = indices;
			// prepare root node
			ref BvhNode root = ref Nodes[ 0 ];
			root.AabbMin = new BvhVec3( BvhConstants.Far );
			root.AabbMax = new BvhVec3( -BvhConstants.Far );
			// prepare fragments
			Fragment[] fragment = Fragments;
			uint[] primIdx = PrimIdx;
			if ( indices == null )
			{
				// building a BVH over triangles specified as three 16-byte vertices each.
				for ( uint i = 0; i < primCount; i++ )
				{
					BvhVec4 v0 = vertices[ i * 3 ], v1 = vertices[ ( i * 3 ) + 1 ], v2 = vertices[ ( i * 3 ) + 2 ];
					BvhVec4 fmin = BvhMath.Min( v0, BvhMath.Min( v1, v2 ) );
					BvhVec4 fmax = BvhMath.Max( v0, BvhMath.Max( v1, v2 ) );
					ref Fragment f = ref fragment[ i ];
					f.BMin = fmin;
					f.BMax = fmax;
					f.PrimIdx = i;
					root.AabbMin = BvhMath.Min( root.AabbMin, f.BMin );
					root.AabbMax = BvhMath.Max( root.AabbMax, f.BMax );
					primIdx[ i ] = i;
				}
			}
			else
			{
				// building a BVH over triangles consisting of vertices indexed by 'indices'.
				for ( uint i = 0; i < primCount; i++ )
				{
					uint i0 = indices[ i * 3 ], i1 = indices[ ( i * 3 ) + 1 ], i2 = indices[ ( i * 3 ) + 2 ];
					BvhVec4 v0 = vertices[ i0 ], v1 = vertices[ i1 ], v2 = vertices[ i2 ];
					BvhVec4 fmin = BvhMath.Min( v0, BvhMath.Min( v1, v2 ) );
					BvhVec4 fmax = BvhMath.Max( v0, BvhMath.Max( v1, v2 ) );
					ref Fragment f = ref fragment[ i ];
					f.BMin = fmin;
					f.BMax = fmax;
					f.PrimIdx = i;
					root.AabbMin = BvhMath.Min( root.AabbMin, f.BMin );
					root.AabbMax = BvhMath.Max( root.AabbMax, f.BMax );
					primIdx[ i ] = i;
				}
			}
			// finalize root node
			root.LeftFirst = 0;
			root.TriCount = primCount;
			// reset node pool
			UsedNodes = 2;
			// all set; actual build happens in BuildBinned / BuildBinnedThreaded.
		}

		/// <summary>TLAS builder: prepares a BVH over the world-space bounds of a list of BLAS instances.</summary>
		private void PrepareTlasBuild( BlasInstance[] instances, uint instCount, Bvh[] blasses, uint bCount )
		{
			TriCount = instCount;
			uint spaceNeeded = instCount * 2; // upper limit
			AllocateNodes( spaceNeeded );
			AllocatePrimIdx( instCount );
			AllocateFragments( instCount );
			Nodes[ 1 ] = default; // node 1 remains unused, for cache line alignment.
			Instances = instances;
			InstanceCount = instCount;
			Blasses = blasses;
			BlasCount = bCount;
			// a TLAS has no vertices of its own; clear any left over from an earlier build.
			Verts = null;
			VertIdx = null;
			// copy relevant data to the fragment array over which the BVH will be built.
			ref BvhNode root = ref Nodes[ 0 ];
			root.LeftFirst = 0;
			root.TriCount = instCount;
			root.AabbMin = new BvhVec3( BvhConstants.Far );
			root.AabbMax = new BvhVec3( -BvhConstants.Far );
			Fragment[] fragment = Fragments;
			uint[] primIdx = PrimIdx;
			for ( uint i = 0; i < instCount; i++ )
			{
				ref BlasInstance inst = ref instances[ i ];
				if ( blasses != null ) // if a null array is passed, we'll assume the instances have been updated elsewhere.
				{
					inst.Update( blasses[ inst.BlasIdx ] );
				}
				ref Fragment f = ref fragment[ i ];
				f.BMin = inst.AabbMin;
				f.PrimIdx = i;
				f.BMax = inst.AabbMax;
				root.AabbMin = BvhMath.Min( root.AabbMin, inst.AabbMin );
				root.AabbMax = BvhMath.Max( root.AabbMax, inst.AabbMax );
				primIdx[ i ] = i;
			}
			// start build
			UsedNodes = 2;
		}

		/// <summary>Reference builder: binned SAH BVH builder. Not using SIMD, serial.</summary>
		private void BuildBinned()
		{
			newNodePtr = ( int )UsedNodes;
			BuildSubtree( 0, 0, null );
			UsedNodes = ( uint )newNodePtr;
			ThreadedSubtrees = 0;
			FinishBuild();
		}

		/// <summary>Shared tail of the serial and the threaded build: the root bounds.</summary>
		private void FinishBuild()
		{
			AabbMin = Nodes[ 0 ].AabbMin;
			AabbMax = Nodes[ 0 ].AabbMax;
		}

		/// <summary>
		/// Port of BVH::Build( nodeIdx, depth ): subdivides nodeIdx and everything below it. Node
		/// pairs are drawn from newNodePtr with an atomic add, so subtrees can run concurrently. When
		/// 'pending' is non-null the walk stops at BvhConstants.MtSpawnDepth: the two children are
		/// recorded there instead of descended into, for the parallel phase to pick up, and the
		/// number of recorded subtree roots is returned. The C++ keeps 'depth' constant inside the
		/// loop because it spawns and returns at every level above the spawn depth; this port
		/// descends instead, so the depth is carried on the task stack.
		/// </summary>
		private int BuildSubtree( uint nodeIdx, uint depth, uint[] pending )
		{
			const int Bins = BvhConstants.Bins;
			BvhNode[] bvhNode = Nodes;
			Fragment[] fragment = Fragments;
			uint[] primIdx = PrimIdx;
			int pendingCount = 0;
			// subdivide the subtree root recursively
			Span<uint> task = stackalloc uint[ 512 ];
			Span<uint> taskDepth = stackalloc uint[ 512 ];
			int taskCount = 0;
			ref BvhNode root = ref bvhNode[ 0 ];
			BvhVec3 minDim = ( root.AabbMax - root.AabbMin ) * 1e-20f;
			BvhVec3 bestLMin = new BvhVec3( 0f ), bestLMax = new BvhVec3( 0f );
			BvhVec3 bestRMin = new BvhVec3( 0f ), bestRMax = new BvhVec3( 0f );
			// scratch for the bins and the per-split totals; the C++ declares these inside the
			// subdivision loop, so they are reset at the start of each iteration below.
			Span<BvhVec3> binMin = stackalloc BvhVec3[ 3 * Bins ];
			Span<BvhVec3> binMax = stackalloc BvhVec3[ 3 * Bins ];
			Span<uint> count = stackalloc uint[ 3 * Bins ];
			Span<BvhVec3> lBMin = stackalloc BvhVec3[ Bins - 1 ];
			Span<BvhVec3> rBMin = stackalloc BvhVec3[ Bins - 1 ];
			Span<BvhVec3> lBMax = stackalloc BvhVec3[ Bins - 1 ];
			Span<BvhVec3> rBMax = stackalloc BvhVec3[ Bins - 1 ];
			Span<float> ANL = stackalloc float[ Bins - 1 ];
			Span<float> ANR = stackalloc float[ Bins - 1 ];
			while ( true )
			{
				while ( true )
				{
					ref BvhNode node = ref bvhNode[ nodeIdx ];
					float SA = node.SurfaceArea;
					if ( SA == 0 )
					{
						break; // can't split an infinitely small node.
					}
					// find optimal object split
					for ( int a = 0; a < 3; a++ )
					{
						for ( int i = 0; i < Bins; i++ )
						{
							binMin[ ( a * Bins ) + i ] = new BvhVec3( BvhConstants.Far );
							binMax[ ( a * Bins ) + i ] = new BvhVec3( -BvhConstants.Far );
							count[ ( a * Bins ) + i ] = 0;
						}
					}
					BvhVec3 extent = node.AabbMax - node.AabbMin;
					BvhVec3 nmin3 = node.AabbMin;
					BvhVec3 rpd3 = new BvhVec3(
						extent.x > minDim.x ? ( Bins / extent.x ) : 0f,
						extent.y > minDim.y ? ( Bins / extent.y ) : 0f,
						extent.z > minDim.z ? ( Bins / extent.z ) : 0f
					);
					for ( uint i = 0; i < node.TriCount; i++ ) // process all tris for x,y and z at once
					{
						ref Fragment f = ref fragment[ primIdx[ node.LeftFirst + i ] ];
						BvhInt3 bi = new BvhInt3( ( ( ( f.BMin + f.BMax ) * 0.5f ) - nmin3 ) * rpd3 );
						bi.x = BvhMath.Clamp( bi.x, 0, Bins - 1 );
						bi.y = BvhMath.Clamp( bi.y, 0, Bins - 1 );
						bi.z = BvhMath.Clamp( bi.z, 0, Bins - 1 );
						binMin[ bi.x ] = BvhMath.Min( binMin[ bi.x ], f.BMin );
						binMax[ bi.x ] = BvhMath.Max( binMax[ bi.x ], f.BMax );
						count[ bi.x ]++;
						binMin[ Bins + bi.y ] = BvhMath.Min( binMin[ Bins + bi.y ], f.BMin );
						binMax[ Bins + bi.y ] = BvhMath.Max( binMax[ Bins + bi.y ], f.BMax );
						count[ Bins + bi.y ]++;
						binMin[ ( 2 * Bins ) + bi.z ] = BvhMath.Min( binMin[ ( 2 * Bins ) + bi.z ], f.BMin );
						binMax[ ( 2 * Bins ) + bi.z ] = BvhMath.Max( binMax[ ( 2 * Bins ) + bi.z ], f.BMax );
						count[ ( 2 * Bins ) + bi.z ]++;
					}
					// calculate per-split totals
					float splitCost = BvhConstants.Far;
					int bestAxis = 0, bestPos = 0;
					for ( int a = 0; a < 3; a++ )
					{
						if ( extent[ a ] > minDim[ a ] )
						{
							BvhVec3 l1 = new BvhVec3( BvhConstants.Far ), l2 = new BvhVec3( -BvhConstants.Far );
							BvhVec3 r1 = new BvhVec3( BvhConstants.Far ), r2 = new BvhVec3( -BvhConstants.Far );
							uint lN = 0, rN = 0;
							for ( int i = 0; i < Bins - 1; i++ )
							{
								lBMin[ i ] = l1 = BvhMath.Min( l1, binMin[ ( a * Bins ) + i ] );
								rBMin[ Bins - 2 - i ] = r1 = BvhMath.Min( r1, binMin[ ( a * Bins ) + ( Bins - 1 - i ) ] );
								lBMax[ i ] = l2 = BvhMath.Max( l2, binMax[ ( a * Bins ) + i ] );
								rBMax[ Bins - 2 - i ] = r2 = BvhMath.Max( r2, binMax[ ( a * Bins ) + ( Bins - 1 - i ) ] );
								lN += count[ ( a * Bins ) + i ];
								rN += count[ ( a * Bins ) + ( Bins - 1 - i ) ];
								ANL[ i ] = lN == 0 ? BvhConstants.Far : ( BvhMath.HalfArea( l2 - l1 ) * ( float )lN );
								ANR[ Bins - 2 - i ] = rN == 0 ? BvhConstants.Far : ( BvhMath.HalfArea( r2 - r1 ) * ( float )rN );
							}
							// evaluate bin totals to find best position for object split
							for ( int i = 0; i < Bins - 1; i++ )
							{
								float C = ANL[ i ] + ANR[ i ];
								if ( C < splitCost )
								{
									splitCost = C;
									bestAxis = a;
									bestPos = i;
									bestLMin = lBMin[ i ];
									bestRMin = rBMin[ i ];
									bestLMax = lBMax[ i ];
									bestRMax = rBMax[ i ];
								}
							}
						}
					}
					splitCost = TraversalCost + ( IntersectionCost * splitCost / SA );
					float noSplitCost = ( float )node.TriCount * IntersectionCost;
					if ( splitCost >= noSplitCost )
					{
						// the C++ prints a warning here when a node with more than 512 prims fails to split.
						break; // not splitting is better.
					}
					// in-place partition
					uint j = node.LeftFirst + node.TriCount, src = node.LeftFirst;
					for ( uint i = 0; i < node.TriCount; i++ )
					{
						// The C++ evaluates a scalar copy of the binning expression here. We reuse the
						// exact vector expression of the binning pass instead, so both passes agree bit
						// for bit even on runtimes that evaluate float arithmetic at higher precision,
						// where a scalar copy can land in a different bin at bin edges.
						// The C++ also casts through uint32_t; the centroid lies inside the node bounds,
						// so the value is never negative and the int cast is equivalent.
						ref Fragment f = ref fragment[ primIdx[ src ] ];
						BvhInt3 bi3 = new BvhInt3( ( ( ( f.BMin + f.BMax ) * 0.5f ) - nmin3 ) * rpd3 );
						int bi = BvhMath.Clamp( bi3[ bestAxis ], 0, Bins - 1 );
						if ( bi <= bestPos )
						{
							src++;
						}
						else
						{
							--j;
							uint t = primIdx[ src ];
							primIdx[ src ] = primIdx[ j ];
							primIdx[ j ] = t;
						}
					}
					// create child nodes
					uint leftCount = src - node.LeftFirst, rightCount = node.TriCount - leftCount;
					if ( leftCount == 0 || rightCount == 0 || taskCount == 512 )
					{
						break; // should not happen.
					}
					uint n = ( uint )( Interlocked.Add( ref newNodePtr, 2 ) - 2 );
					ref BvhNode left = ref bvhNode[ n ];
					left.AabbMin = bestLMin;
					left.AabbMax = bestLMax;
					left.LeftFirst = node.LeftFirst;
					left.TriCount = leftCount;
					ref BvhNode right = ref bvhNode[ n + 1 ];
					right.AabbMin = bestRMin;
					right.AabbMax = bestRMax;
					right.LeftFirst = j;
					right.TriCount = rightCount;
					node.LeftFirst = n;
					node.TriCount = 0;
					if ( pending != null && ( depth + 1 ) == BvhConstants.MtSpawnDepth )
					{
						// hand both children to the parallel phase instead of descending into them.
						pending[ pendingCount++ ] = n;
						pending[ pendingCount++ ] = n + 1;
						break;
					}
					task[ taskCount ] = n + 1;
					taskDepth[ taskCount++ ] = depth + 1;
					nodeIdx = n;
					depth++;
				}
				// fetch subdivision task from stack
				if ( taskCount == 0 )
				{
					break;
				}
				taskCount--;
				nodeIdx = task[ taskCount ];
				depth = taskDepth[ taskCount ];
			}
			return pendingCount;
		}
	}

	// ============================================================================
	//
	//        T H R E A D E D   C O N S T R U C T I O N
	//
	// ============================================================================

	/// <summary>
	/// Threaded binned construction, Parallel.For standing in for tinybvh's tinybvh_spawn /
	/// tinybvh_barrier hooks. The build runs in two phases: the calling thread subdivides the top of
	/// the tree down to BvhConstants.MtSpawnDepth and collects the subtrees rooted there, then
	/// Parallel.For builds those subtrees, which touch disjoint primitive index ranges and draw their
	/// nodes from the shared atomic counter. That matches the C++, where the same subtrees end up on
	/// the thread pool; only the order in which node pairs are handed out differs, so the node
	/// numbering of a threaded build is not reproducible while the tree shape is. Entered from
	/// RunBinnedBuild once the primitive count reaches BvhConstants.MtBuildThreshold.
	/// </summary>
	public sealed partial class Bvh
	{
		/// <summary>Threaded binned SAH build; the threaded counterpart of BuildBinned.</summary>
		private void BuildBinnedThreaded()
		{
			// upper bound on the collected subtrees: a binary tree has at most 2^N nodes at depth N.
			const int MaxSubtrees = 1 << BvhConstants.MtSpawnDepth;
			uint[] pending = new uint[ MaxSubtrees ];
			// newNodePtr is the node allocator shared by both phases.
			newNodePtr = ( int )UsedNodes;
			int subtrees = BuildSubtree( 0, 0, pending );
			if ( subtrees > 0 )
			{
				Parallel.For( 0, subtrees, i => BuildSubtree( pending[ i ], BvhConstants.MtSpawnDepth, null ) );
			}
			UsedNodes = ( uint )newNodePtr;
			ThreadedSubtrees = ( uint )subtrees;
			FinishBuild();
		}
	}

	// ============================================================================
	//
	//        C U S T O M   G E O M E T R Y   B U I L D S
	//
	// ============================================================================

	/// <summary>
	/// BVHs over custom geometry: the tree is built over user-supplied AABBs and traversal hands
	/// the primitive indices in a leaf to the CustomIntersect / CustomIsOccluded callbacks.
	/// Port of BVH::BuildAABB and BVH::Build( customGetAABB, primCount ); both fill the fragment
	/// array and then run the binned SAH reference builder. There is no vertex array to refit such
	/// a tree to, so rebuild it when the primitives move.
	/// </summary>
	public sealed partial class Bvh
	{
		/// <summary>
		/// Builds over a list of AABBs: two vectors per primitive, min then max; only xyz is used.
		/// The AABBs are not referenced after the build.
		/// </summary>
		public void BuildAabbs( BvhVec4[] aabbs, uint primCount )
		{
			ValidateCustomBuildInput( primCount );
			if ( aabbs == null )
			{
				throw new ArgumentException( "Bvh.BuildAabbs( .. ), aabbs == null.", nameof( aabbs ) );
			}
			if ( aabbs.Length < ( long )primCount * 2 )
			{
				throw new ArgumentException( "Bvh.BuildAabbs( .. ), aabbs holds fewer than two vectors per primitive.", nameof( aabbs ) );
			}
			PrepareAabbBuild( aabbs, primCount );
			RunBinnedBuild();
		}

		/// <summary>Builds over custom geometry: the bounds of every primitive are obtained from a callback.</summary>
		public void Build( GetAabbDelegate getAabb, uint primCount )
		{
			ValidateCustomBuildInput( primCount );
			if ( getAabb == null )
			{
				throw new ArgumentException( "Bvh.Build( .. ), getAabb == null.", nameof( getAabb ) );
			}
			PrepareCustomBuild( getAabb, primCount );
			RunBinnedBuild();
		}

		/// <summary>Managed-side replacement for the BVH_FATAL_ERROR checks of the custom builders.</summary>
		private static void ValidateCustomBuildInput( uint primCount )
		{
			if ( primCount == 0 )
			{
				throw new ArgumentException( "Bvh.BuildAabbs( .. ), primCount == 0.", nameof( primCount ) );
			}
		}

		/// <summary>Port of the allocation and fragment loop of BVH::BuildAABB.</summary>
		private void PrepareAabbBuild( BvhVec4[] aabbs, uint primCount )
		{
			ref BvhNode root = ref PrepareCustomRoot( primCount );
			Fragment[] fragment = Fragments;
			uint[] primIdx = PrimIdx;
			for ( uint i = 0; i < primCount; i++ )
			{
				ref Fragment f = ref fragment[ i ];
				f.BMin = aabbs[ i * 2 ];
				f.BMax = aabbs[ ( i * 2 ) + 1 ];
				f.PrimIdx = i;
				primIdx[ i ] = i;
				root.AabbMin = BvhMath.Min( root.AabbMin, f.BMin );
				root.AabbMax = BvhMath.Max( root.AabbMax, f.BMax );
			}
		}

		/// <summary>Port of the allocation and fragment loop of BVH::Build( customGetAABB, primCount ).</summary>
		private void PrepareCustomBuild( GetAabbDelegate getAabb, uint primCount )
		{
			ref BvhNode root = ref PrepareCustomRoot( primCount );
			Fragment[] fragment = Fragments;
			uint[] primIdx = PrimIdx;
			for ( uint i = 0; i < primCount; i++ )
			{
				ref Fragment f = ref fragment[ i ];
				getAabb( i, out f.BMin, out f.BMax );
				f.PrimIdx = i;
				primIdx[ i ] = i;
				root.AabbMin = BvhMath.Min( root.AabbMin, f.BMin );
				root.AabbMax = BvhMath.Max( root.AabbMax, f.BMax );
			}
		}

		/// <summary>Shared allocation and root setup of both custom builders; returns the root node.</summary>
		private ref BvhNode PrepareCustomRoot( uint primCount )
		{
			TriCount = primCount;
			uint spaceNeeded = primCount * 2; // upper limit
			AllocateNodes( spaceNeeded );
			AllocatePrimIdx( primCount );
			AllocateFragments( primCount );
			Nodes[ 1 ] = default; // node 1 remains unused, for cache line alignment.
			// there is no vertex data: leaf primitives are handed to the custom callbacks. Clear any
			// left over from an earlier build, as PrepareTlasBuild does.
			Verts = null;
			VertIdx = null;
			ref BvhNode root = ref Nodes[ 0 ];
			root.LeftFirst = 0;
			root.TriCount = primCount;
			root.AabbMin = new BvhVec3( BvhConstants.Far );
			root.AabbMax = new BvhVec3( -BvhConstants.Far );
			// start build
			UsedNodes = 2;
			return ref root;
		}
	}

	// ============================================================================
	//
	//        B V H   T R A V E R S A L
	//
	// ============================================================================

	/// <summary>
	/// Traversal half of tinybvh's BVH class. The C++ compiles eight template variants of each
	/// traversal function, keyed on the sign of the ray direction (posX, posY, posZ); here the
	/// same predicates are evaluated once at the top of the traversal and passed down, so the
	/// arithmetic - and therefore the resulting t values - stays identical. The traversal stacks
	/// hold node indices where the C++ stacks hold node pointers. Traversal keeps no state in the
	/// BVH, so any number of threads can trace against the same tree at once.
	/// </summary>
	public sealed partial class Bvh
	{
		/// <summary>Traversal stack depth of BVH::Intersect (C++: BVHNode* stack[256]).</summary>
		private const int IntersectStackSize = 256;
		/// <summary>Traversal stack depth of IntersectTLAS, IsOccluded and IsOccludedTLAS (C++: stack[64]).</summary>
		private const int SmallStackSize = 64;

		/// <summary>
		/// Port of BVH::Intersect( Ray&amp; ), including its octant dispatcher. Returns the traversal
		/// cost; the hit, if any, is written to ray.Hit. A BVH that was never built reports no hit.
		/// </summary>
		public int Intersect( ref Ray ray )
		{
			if ( UsedNodes == 0 )
			{
				return 0;
			}
			bool posX = ray.D.x >= 0f;
			bool posY = ray.D.y >= 0f;
			bool posZ = ray.D.z >= 0f;
			if ( IsTlas )
			{
				return IntersectTlas( ref ray, posX, posY, posZ );
			}
			return IntersectBlas( ref ray, posX, posY, posZ );
		}

		/// <summary>
		/// Port of BVH::IsOccluded( const Ray&amp; ), including its octant dispatcher. Returns true as
		/// soon as any primitive is hit within ray.Hit.T; the ray itself is left untouched.
		/// </summary>
		public bool IsOccluded( in Ray ray )
		{
			if ( UsedNodes == 0 )
			{
				return false;
			}
			bool posX = ray.D.x >= 0f;
			bool posY = ray.D.y >= 0f;
			bool posZ = ray.D.z >= 0f;
			if ( IsTlas )
			{
				return IsOccludedTlas( ray, posX, posY, posZ );
			}
			return IsOccludedBlas( ray, posX, posY, posZ );
		}

		/// <summary>Port of the templated BVH::Intersect body: traversal over triangles.</summary>
		private int IntersectBlas( ref Ray ray, bool posX, bool posY, bool posZ )
		{
			BvhNode[] nodes = Nodes;
			uint[] primIdx = PrimIdx;
			BvhVec4[] verts = Verts;
			Span<uint> stack = stackalloc uint[ IntersectStackSize ];
			int stackPtr = 0;
			uint nodeIdx = 0;
			float cost = 0f;
			float rox = ray.O.x * ray.RD.x;
			float roy = ray.O.y * ray.RD.y;
			float roz = ray.O.z * ray.RD.z;
			while ( true )
			{
				cost += TraversalCost;
				ref BvhNode node = ref nodes[ nodeIdx ];
				if ( node.IsLeaf )
				{
					// Performance note: if indexed primitives (ENABLE_INDEXED_GEOMETRY) and custom
					// geometry (ENABLE_CUSTOM_GEOMETRY) are both disabled, this leaf code reduces
					// to a regular loop over triangles. Otherwise, the extra flexibility comes at
					// a small performance cost.
					// The C++ picks between three leaf loops: indexed, custom and plain triangles.
					// GetPrimIndices already covers indexed and plain, so one test of the callback
					// per leaf is left. Unlike the C++, which tries the indexed loop first, a BVH
					// that has both VertIdx and a callback set uses the callback here.
					if ( CustomIntersect != null )
					{
						for ( uint i = 0; i < node.TriCount; i++ )
						{
							if ( CustomIntersect( ref ray, primIdx[ node.LeftFirst + i ] ) )
							{
								// INST_IDX_BITS == 32: the instance index lives in its own field.
								ray.Hit.Inst = ray.InstIdx;
							}
							cost += IntersectionCost;
						}
					}
					else
					{
						for ( uint i = 0; i < node.TriCount; i++ )
						{
							uint pi = primIdx[ node.LeftFirst + i ];
							GetPrimIndices( pi, out uint i0, out uint i1, out uint i2 );
							IntersectTri( ref ray, pi, verts, i0, i1, i2 );
							cost += IntersectionCost;
						}
					}
					if ( stackPtr == 0 )
					{
						break;
					}
					nodeIdx = stack[ --stackPtr ];
					continue;
				}
				uint child1 = node.LeftFirst, child2 = node.LeftFirst + 1;
				SlabTestTwoNodes( nodes[ child1 ], nodes[ child2 ], ray, rox, roy, roz, posX, posY, posZ, out float dist1, out float dist2 );
				if ( dist1 > dist2 )
				{
					BvhMath.Swap( ref dist1, ref dist2 );
					BvhMath.Swap( ref child1, ref child2 );
				}
				if ( dist1 == BvhConstants.Far /* missed both child nodes */ )
				{
					if ( stackPtr == 0 )
					{
						break;
					}
					nodeIdx = stack[ --stackPtr ];
				}
				else /* hit at least one node */
				{
					nodeIdx = child1; /* continue with the nearest */
					if ( dist2 != BvhConstants.Far )
					{
						stack[ stackPtr++ ] = child2; /* push far child */
					}
				}
			}
			return ( int )cost; // cast to not break interface.
		}

		/// <summary>
		/// Port of the templated BVH::IntersectTLAS body: traversal over BLAS instances. Each BLAS
		/// is entered through its public Intersect, as in the C++, which recomputes the octant
		/// predicates for the transformed direction.
		/// </summary>
		private int IntersectTlas( ref Ray ray, bool posX, bool posY, bool posZ )
		{
			BvhNode[] nodes = Nodes;
			uint[] primIdx = PrimIdx;
			BlasInstance[] instances = Instances;
			Bvh[] blasses = Blasses;
			Span<uint> stack = stackalloc uint[ SmallStackSize ];
			int stackPtr = 0;
			uint nodeIdx = 0;
			float cost = 0f;
			float rox = ray.O.x * ray.RD.x;
			float roy = ray.O.y * ray.RD.y;
			float roz = ray.O.z * ray.RD.z;
			while ( true )
			{
				cost += TraversalCost;
				ref BvhNode node = ref nodes[ nodeIdx ];
				if ( node.IsLeaf )
				{
					// the C++ Ray member initializers: mask = RAY_MASK_INTERSECT_ALL, instIdx = 0.
					Ray tmpRay = default;
					tmpRay.Mask = BvhConstants.RayMaskIntersectAll;
					for ( uint i = 0; i < node.TriCount; i++ )
					{
						// BLAS traversal
						uint instIdx = primIdx[ node.LeftFirst + i ];
						ref BlasInstance inst = ref instances[ instIdx ];
						// Check if the ray should intersect this BLAS Instance, otherwise skip it
						if ( ( inst.Mask & ray.Mask ) == 0 )
						{
							continue;
						}
						// 1. Transform ray with the inverse of the instance transform
						tmpRay.O = inst.InvTransform.TransformPoint( ray.O );
						tmpRay.D = inst.InvTransform.TransformVector( ray.D );
						tmpRay.InstIdx = instIdx; // INST_IDX_BITS == 32, so the C++ shift by (32 - 32) is a no-op.
						tmpRay.Hit = ray.Hit;
						tmpRay.RD = BvhMath.Rcp( tmpRay.D );
						// 2. Traverse BLAS with the transformed ray
						cost += blasses[ inst.BlasIdx ].Intersect( ref tmpRay );
						// 3. Restore ray
						ray.Hit = tmpRay.Hit;
					}
					if ( stackPtr == 0 )
					{
						break;
					}
					nodeIdx = stack[ --stackPtr ];
					continue;
				}
				uint child1 = node.LeftFirst, child2 = node.LeftFirst + 1;
				SlabTestTwoNodes( nodes[ child1 ], nodes[ child2 ], ray, rox, roy, roz, posX, posY, posZ, out float dist1, out float dist2 );
				if ( dist1 > dist2 )
				{
					BvhMath.Swap( ref dist1, ref dist2 );
					BvhMath.Swap( ref child1, ref child2 );
				}
				if ( dist1 == BvhConstants.Far /* missed both child nodes */ )
				{
					if ( stackPtr == 0 )
					{
						break;
					}
					nodeIdx = stack[ --stackPtr ];
				}
				else /* hit at least one node */
				{
					nodeIdx = child1; /* continue with the nearest */
					if ( dist2 != BvhConstants.Far )
					{
						stack[ stackPtr++ ] = child2; /* push far child */
					}
				}
			}
			return ( int )cost;
		}

		/// <summary>Port of the templated BVH::IsOccluded body.</summary>
		private bool IsOccludedBlas( in Ray ray, bool posX, bool posY, bool posZ )
		{
			BvhNode[] nodes = Nodes;
			uint[] primIdx = PrimIdx;
			BvhVec4[] verts = Verts;
			Span<uint> stack = stackalloc uint[ SmallStackSize ];
			int stackPtr = 0;
			uint nodeIdx = 0;
			float rox = ray.O.x * ray.RD.x;
			float roy = ray.O.y * ray.RD.y;
			float roz = ray.O.z * ray.RD.z;
			while ( true )
			{
				ref BvhNode node = ref nodes[ nodeIdx ];
				if ( node.IsLeaf )
				{
					// See the note in IntersectBlas: one test of the callback per leaf.
					if ( CustomIsOccluded != null )
					{
						for ( uint i = 0; i < node.TriCount; i++ )
						{
							if ( CustomIsOccluded( ray, primIdx[ node.LeftFirst + i ] ) )
							{
								return true;
							}
						}
					}
					else
					{
						for ( uint i = 0; i < node.TriCount; i++ )
						{
							uint pi = primIdx[ node.LeftFirst + i ];
							GetPrimIndices( pi, out uint i0, out uint i1, out uint i2 );
							if ( TriOccludes( ray, verts, i0, i1, i2 ) )
							{
								return true;
							}
						}
					}
					if ( stackPtr == 0 )
					{
						break;
					}
					nodeIdx = stack[ --stackPtr ];
					continue;
				}
				uint child1 = node.LeftFirst, child2 = node.LeftFirst + 1;
				SlabTestTwoNodes( nodes[ child1 ], nodes[ child2 ], ray, rox, roy, roz, posX, posY, posZ, out float dist1, out float dist2 );
				if ( dist1 > dist2 )
				{
					BvhMath.Swap( ref dist1, ref dist2 );
					BvhMath.Swap( ref child1, ref child2 );
				}
				if ( dist1 == BvhConstants.Far /* missed both child nodes */ )
				{
					if ( stackPtr == 0 )
					{
						break;
					}
					nodeIdx = stack[ --stackPtr ];
				}
				else /* hit at least one node */
				{
					nodeIdx = child1; /* continue with the nearest */
					if ( dist2 != BvhConstants.Far )
					{
						stack[ stackPtr++ ] = child2; /* push far child */
					}
				}
			}
			return false;
		}

		/// <summary>Port of the templated BVH::IsOccludedTLAS body; see <see cref="IntersectTlas"/>.</summary>
		private bool IsOccludedTlas( in Ray ray, bool posX, bool posY, bool posZ )
		{
			BvhNode[] nodes = Nodes;
			uint[] primIdx = PrimIdx;
			BlasInstance[] instances = Instances;
			Bvh[] blasses = Blasses;
			Span<uint> stack = stackalloc uint[ SmallStackSize ];
			int stackPtr = 0;
			uint nodeIdx = 0;
			// the C++ Ray member initializers: mask = RAY_MASK_INTERSECT_ALL, instIdx = 0.
			Ray tmpRay = default;
			tmpRay.Mask = BvhConstants.RayMaskIntersectAll;
			float rox = ray.O.x * ray.RD.x;
			float roy = ray.O.y * ray.RD.y;
			float roz = ray.O.z * ray.RD.z;
			while ( true )
			{
				ref BvhNode node = ref nodes[ nodeIdx ];
				if ( node.IsLeaf )
				{
					for ( uint i = 0; i < node.TriCount; i++ )
					{
						// BLAS traversal
						ref BlasInstance inst = ref instances[ primIdx[ node.LeftFirst + i ] ];
						// Check if the ray should intersect this BLAS Instance, otherwise skip it
						if ( ( inst.Mask & ray.Mask ) == 0 )
						{
							continue;
						}
						// 1. Transform ray with the inverse of the instance transform
						tmpRay.O = inst.InvTransform.TransformPoint( ray.O );
						tmpRay.D = inst.InvTransform.TransformVector( ray.D );
						tmpRay.Hit.T = ray.Hit.T;
						tmpRay.RD = BvhMath.Rcp( tmpRay.D );
						// 2. Traverse BLAS with the transformed ray
						if ( blasses[ inst.BlasIdx ].IsOccluded( tmpRay ) )
						{
							return true;
						}
					}
					if ( stackPtr == 0 )
					{
						break;
					}
					nodeIdx = stack[ --stackPtr ];
					continue;
				}
				uint child1 = node.LeftFirst, child2 = node.LeftFirst + 1;
				SlabTestTwoNodes( nodes[ child1 ], nodes[ child2 ], ray, rox, roy, roz, posX, posY, posZ, out float dist1, out float dist2 );
				if ( dist1 > dist2 )
				{
					BvhMath.Swap( ref dist1, ref dist2 );
					BvhMath.Swap( ref child1, ref child2 );
				}
				if ( dist1 == BvhConstants.Far /* missed both child nodes */ )
				{
					if ( stackPtr == 0 )
					{
						break;
					}
					nodeIdx = stack[ --stackPtr ];
				}
				else /* hit at least one node */
				{
					nodeIdx = child1; /* continue with the nearest */
					if ( dist2 != BvhConstants.Far )
					{
						stack[ stackPtr++ ] = child2; /* push far child */
					}
				}
			}
			return false;
		}

		/// <summary>
		/// Port of the SLAB_TEST_TWO_NODES macro: slab test against both children of a node at once.
		/// The octant flags select which corner of each box yields the near plane, so no min/max
		/// per axis is needed. Outputs BvhConstants.Far for a child that was missed.
		/// </summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static void SlabTestTwoNodes( in BvhNode child1, in BvhNode child2, in Ray ray,
			float rox, float roy, float roz, bool posX, bool posY, bool posZ,
			out float dist1, out float dist2 )
		{
			float tx1a = ( ( posX ? child1.AabbMin.x : child1.AabbMax.x ) * ray.RD.x ) - rox; /* expect fma. */
			float ty1a = ( ( posY ? child1.AabbMin.y : child1.AabbMax.y ) * ray.RD.y ) - roy;
			float tz1a = ( ( posZ ? child1.AabbMin.z : child1.AabbMax.z ) * ray.RD.z ) - roz;
			float tx1b = ( ( posX ? child2.AabbMin.x : child2.AabbMax.x ) * ray.RD.x ) - rox;
			float ty1b = ( ( posY ? child2.AabbMin.y : child2.AabbMax.y ) * ray.RD.y ) - roy;
			float tz1b = ( ( posZ ? child2.AabbMin.z : child2.AabbMax.z ) * ray.RD.z ) - roz;
			float tx2a = ( ( posX ? child1.AabbMax.x : child1.AabbMin.x ) * ray.RD.x ) - rox;
			float ty2a = ( ( posY ? child1.AabbMax.y : child1.AabbMin.y ) * ray.RD.y ) - roy;
			float tz2a = ( ( posZ ? child1.AabbMax.z : child1.AabbMin.z ) * ray.RD.z ) - roz;
			float tx2b = ( ( posX ? child2.AabbMax.x : child2.AabbMin.x ) * ray.RD.x ) - rox;
			float ty2b = ( ( posY ? child2.AabbMax.y : child2.AabbMin.y ) * ray.RD.y ) - roy;
			float tz2b = ( ( posZ ? child2.AabbMax.z : child2.AabbMin.z ) * ray.RD.z ) - roz;
			float tmina = BvhMath.Max( BvhMath.Max( tx1a, ty1a ), BvhMath.Max( tz1a, 0.0f ) );
			float tminb = BvhMath.Max( BvhMath.Max( tx1b, ty1b ), BvhMath.Max( tz1b, 0.0f ) );
			float tmaxa = BvhMath.Min( BvhMath.Min( tx2a, ty2a ), BvhMath.Min( tz2a, ray.Hit.T ) );
			float tmaxb = BvhMath.Min( BvhMath.Min( tx2b, ty2b ), BvhMath.Min( tz2b, ray.Hit.T ) );
			dist1 = BvhConstants.Far;
			dist2 = BvhConstants.Far;
			if ( tmaxa >= tmina )
			{
				dist1 = tmina;
			}
			if ( tmaxb >= tminb )
			{
				dist2 = tminb;
			}
		}

		/// <summary>
		/// Port of BVHBase::IntersectTri, Moeller-Trumbore path. Registers a hit only when it is
		/// closer than the current ray.Hit.T.
		/// </summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static void IntersectTri( ref Ray ray, uint triIdx, BvhVec4[] verts, uint i0, uint i1, uint i2 )
		{
			// Moeller-Trumbore ray/triangle intersection algorithm.
			BvhVec4 v0_ = verts[ i0 ];
			BvhVec3 v0 = v0_;
			BvhVec3 e1 = verts[ i1 ] - v0_;
			BvhVec3 e2 = verts[ i2 ] - v0_;
			BvhVec3 h = BvhMath.Cross( ray.D, e2 );
			float a = BvhMath.Dot( e1, h );
			if ( MathF.Abs( a ) < 0.000001f )
			{
				return;
			}
			float f = 1f / a;
			BvhVec3 s = ray.O - v0;
			float u = f * BvhMath.Dot( s, h );
			BvhVec3 q = BvhMath.Cross( s, e1 );
			float v = f * BvhMath.Dot( ray.D, q );
			bool miss = u < 0f || v < 0f || ( u + v ) > 1f;
			if ( miss )
			{
				return;
			}
			float t = f * BvhMath.Dot( e2, q );
			if ( t < 0f || t > ray.Hit.T )
			{
				return;
			}
			// register a hit: ray is shortened to t.
			ray.Hit.T = t;
			ray.Hit.U = u;
			ray.Hit.V = v;
			// INST_IDX_BITS == 32: the instance index lives in its own field.
			ray.Hit.Prim = triIdx;
			ray.Hit.Inst = ray.InstIdx;
		}

		/// <summary>
		/// Port of BVHBase::TriOccludes, Moeller-Trumbore path. ray.Hit.T is the maximum distance;
		/// the ray is not modified.
		/// </summary>
		[MethodImpl( MethodImplOptions.AggressiveInlining )]
		private static bool TriOccludes( in Ray ray, BvhVec4[] verts, uint i0, uint i1, uint i2 )
		{
			// Moeller-Trumbore ray/triangle intersection algorithm
			BvhVec4 v0_ = verts[ i0 ];
			BvhVec3 v0 = v0_;
			BvhVec3 e1 = verts[ i1 ] - v0_;
			BvhVec3 e2 = verts[ i2 ] - v0_;
			BvhVec3 h = BvhMath.Cross( ray.D, e2 );
			float a = BvhMath.Dot( e1, h );
			if ( MathF.Abs( a ) < 0.000001f )
			{
				return false;
			}
			float f = 1f / a;
			BvhVec3 s = ray.O - v0;
			float u = f * BvhMath.Dot( s, h );
			BvhVec3 q = BvhMath.Cross( s, e1 );
			float v = f * BvhMath.Dot( ray.D, q );
			bool miss = u < 0f || v < 0f || ( u + v ) > 1f;
			if ( miss )
			{
				return false;
			}
			float t = f * BvhMath.Dot( e2, q );
			if ( t < 0f || t > ray.Hit.T )
			{
				return false;
			}
			// occluded.
			return true;
		}
	}
}
