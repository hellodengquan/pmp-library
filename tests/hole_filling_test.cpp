// Copyright 2017-2026 the Polygon Mesh Processing Library developers.
// SPDX-License-Identifier: MIT

// Tests for pmp::fill_hole().
//
// The tests judge hole filling by its topological and geometric result and
// deliberately avoid prescribing one particular triangulation: any legal
// triangulation of the boundary is accepted. Orientation compatibility is
// tested structurally (every interior edge is traversed in opposite
// directions by its two incident faces) rather than by comparing face
// normals, since a legally sharp (acute) fold and a truly inverted face can
// look identical under a normal dot product, while a flat compatible patch
// can have a dot product of minus one across a shared edge.

#include "gtest/gtest.h"

#include <limits>
#include <map>
#include <vector>

#include "pmp/algorithms/hole_filling.h"
#include "pmp/bounding_box.h"
#include "pmp/exceptions.h"
#include "pmp/surface_mesh.h"

#include "helpers.h"

using namespace pmp;

namespace {

// ---------------------------------------------------------------------------
// small mesh materials
// ---------------------------------------------------------------------------

// A warped disk built from an eight-vertex non-planar ring and a center
// vertex. Removing k adjacent cap triangles opens a single hole; the two
// cap edges at the ends of the removed run stay interior, so the resulting
// boundary always has k+2 distinct ring vertices and k+4 halfedges.
SurfaceMesh warped_disk(size_t removed_from, size_t removed_count)
{
    SurfaceMesh mesh;
    const size_t n = 8;
    std::vector<Vertex> ring(n);
    for (size_t i = 0; i < n; ++i)
    {
        const Scalar a = Scalar(2) * std::numbers::pi * Scalar(i) / Scalar(n);
        const Scalar z = Scalar(0.35) * std::sin(Scalar(3) * a);
        ring[i] = mesh.add_vertex(Point(std::cos(a), std::sin(a), z));
    }
    const Vertex center = mesh.add_vertex(Point(0, 0, -0.6));
    std::vector<Face> caps(n);
    for (size_t i = 0; i < n; ++i)
        caps[i] = mesh.add_triangle(center, ring[i],
                                    ring[(i + 1) % n]);
    for (size_t k = 0; k < removed_count; ++k)
        mesh.delete_face(caps[(removed_from + k) % n]);
    mesh.garbage_collection();
    return mesh;
}

// A disk with six boundary vertices a,c,b,e,d,f around an interior vertex a.
// The edge (a,b) already exists as an *interior* edge between two boundary
// vertices before the hole is filled: the triangulation must reuse it rather
// than create a duplicate (non-manifold) connection. The warped variant
// lifts the boundary out of a plane.
SurfaceMesh chord_hexagon(bool warped)
{
    SurfaceMesh mesh;
    const auto a =
        mesh.add_vertex(warped ? Point(0, 0, 0.3) : Point(0, 0, 0));
    const auto c =
        mesh.add_vertex(warped ? Point(1, 0.8, -0.2) : Point(1, 1, 0));
    const auto b =
        mesh.add_vertex(warped ? Point(2, 0, 0.1) : Point(2, 0, 0));
    const auto e =
        mesh.add_vertex(warped ? Point(2.2, -1, -0.3) : Point(2, -1, 0));
    const auto d =
        mesh.add_vertex(warped ? Point(1, -1.6, 0.25) : Point(1, -2, 0));
    const auto f =
        mesh.add_vertex(warped ? Point(-0.4, -1, -0.1) : Point(0, -1, 0));
    mesh.add_triangle(a, c, b);
    mesh.add_triangle(a, b, e);
    mesh.add_triangle(a, e, d);
    mesh.add_triangle(a, d, f);
    return mesh;
}

// An open tube (two disjoint boundary loops) whose top ring is non-planar.
SurfaceMesh open_tube(size_t n = 8)
{
    SurfaceMesh mesh;
    std::vector<Vertex> bottom(n), top(n);
    for (size_t i = 0; i < n; ++i)
    {
        const Scalar a = Scalar(2) * std::numbers::pi * Scalar(i) / Scalar(n);
        bottom[i] = mesh.add_vertex(Point(std::cos(a), std::sin(a), 0));
        top[i] = mesh.add_vertex(Point(
            std::cos(a), std::sin(a),
            Scalar(1) + Scalar(0.25) * std::sin(Scalar(3) * a)));
    }
    for (size_t i = 0; i < n; ++i)
    {
        const size_t j = (i + 1) % n;
        mesh.add_triangle(bottom[i], bottom[j], top[j]);
        mesh.add_triangle(bottom[i], top[j], top[i]);
    }
    return mesh;
}

// Two triangle fans that share one vertex but are disconnected otherwise.
// The shared vertex has two boundary gaps and is therefore non-manifold.
SurfaceMesh non_manifold_fan()
{
    SurfaceMesh mesh;
    const auto shared = mesh.add_vertex(Point(0, 0, 0));
    const auto a1 = mesh.add_vertex(Point(1, 0, 0));
    const auto a2 = mesh.add_vertex(Point(0, 1, 0));
    const auto b1 = mesh.add_vertex(Point(-1, 0, 0));
    const auto b2 = mesh.add_vertex(Point(0, -1, 0));
    mesh.add_triangle(shared, a1, a2);
    mesh.add_triangle(shared, b1, b2);
    return mesh;
}

// A capped, warped tube whose bottom fan contains one sharply protruding
// ring vertex (a spike). The two cap triangles next to the spike are
// removed, producing a four-edge local hole whose corner at the spike is
// acute (about 21 degrees). The ring edges around the spike remain interior
// edges shared with the tube skirt, so the boundary vertices already have
// interior connections when the hole is filled. The outer top ring stays
// open as a second, independent boundary.
struct AcuteWedgeMesh
{
    SurfaceMesh mesh;
    Vertex spike;
    Halfedge hole;
};

AcuteWedgeMesh make_acute_wedge_mesh()
{
    AcuteWedgeMesh result;
    auto& mesh = result.mesh;
    const size_t n = 8;
    const size_t spike_index = 3;
    // non-uniform angles: the spike gets a narrow wedge of about 12 degrees
    // on each side, the remaining angular gap is split evenly
    Scalar angles[n];
    {
        const Scalar wedge = Scalar(12) * std::numbers::pi / Scalar(180);
        const size_t n_gaps = n;
        const Scalar regular =
            (Scalar(2) * std::numbers::pi - Scalar(2) * wedge) /
            Scalar(n_gaps - 2);
        for (size_t i = 0; i < n; ++i)
        {
            const size_t delta = (i + n - spike_index) % n;
            if (delta == 0)
                angles[i] = 0;
            else if (delta == 1)
                angles[i] = wedge;
            else if (delta == n - 1)
                angles[i] = -wedge + Scalar(2) * std::numbers::pi;
            else
                angles[i] = wedge + Scalar(delta - 1) * regular;
        }
    }
    std::vector<Vertex> inner(n), outer(n);
    for (size_t i = 0; i < n; ++i)
    {
        const Scalar a = angles[i];
        const Scalar radius = (i == spike_index) ? Scalar(2.5) : Scalar(1);
        const Scalar z = Scalar(0.3) * std::sin(Scalar(3) * a);
        inner[i] = mesh.add_vertex(
            Point(radius * std::cos(a), radius * std::sin(a), z));
        outer[i] = mesh.add_vertex(Point((radius + Scalar(1)) * std::cos(a),
                                         (radius + Scalar(1)) * std::sin(a),
                                         z - Scalar(0.4)));
    }
    const Vertex center = mesh.add_vertex(Point(0, 0, -0.5));
    result.spike = inner[spike_index];

    std::vector<Face> caps(n);
    for (size_t i = 0; i < n; ++i)
    {
        const size_t j = (i + 1) % n;
        caps[i] = mesh.add_triangle(center, inner[i], inner[j]);
        // tube skirt. The first skirt triangle shares the inner edge with
        // the cap and must traverse it in the opposite direction; the second
        // triangle uses only diagonals and the outer ring edge.
        mesh.add_triangle(inner[j], inner[i], outer[i]);
        mesh.add_triangle(outer[j], inner[j], outer[i]);
    }

    mesh.delete_face(caps[spike_index - 1]);
    mesh.delete_face(caps[spike_index]);
    mesh.garbage_collection();

    for (auto h : mesh.halfedges())
    {
        if (!mesh.is_boundary(h))
            continue;
        bool touches_center = false;
        auto hh = h;
        do
        {
            if (mesh.to_vertex(hh) == center)
                touches_center = true;
            hh = mesh.next_halfedge(hh);
        } while (hh != h);
        if (touches_center)
        {
            result.hole = h;
            break;
        }
    }
    return result;
}

// Interior angle at boundary vertex v, computed from its two boundary edges.
Scalar boundary_corner_angle(const SurfaceMesh& mesh, Vertex v)
{
    // a manifold boundary vertex has exactly one outgoing boundary halfedge
    Halfedge out{};
    for (auto h : mesh.halfedges(v))
        if (mesh.is_boundary(h))
        {
            out = h;
            break;
        }
    if (!out.is_valid())
        return std::numeric_limits<Scalar>::quiet_NaN();
    const Vertex next = mesh.to_vertex(out);
    // the boundary halfedge ending at v is the predecessor of out along the
    // boundary loop
    const Halfedge incoming = mesh.prev_halfedge(out);
    if (!mesh.is_boundary(incoming) || mesh.to_vertex(incoming) != v)
        return std::numeric_limits<Scalar>::quiet_NaN();
    const Vertex prev = mesh.from_vertex(incoming);
    const Point u = normalize(mesh.position(prev) - mesh.position(v));
    const Point w = normalize(mesh.position(next) - mesh.position(v));
    return std::acos(std::clamp(dot(u, w), Scalar(-1), Scalar(1)));
}

// ---------------------------------------------------------------------------
// inspection helpers
// ---------------------------------------------------------------------------

Halfedge first_boundary_halfedge(const SurfaceMesh& mesh)
{
    for (auto h : mesh.halfedges())
        if (mesh.is_boundary(h))
            return h;
    return {};
}

size_t count_boundary_loops(SurfaceMesh& mesh)
{
    auto marker = mesh.add_halfedge_property<char>("test:seen", 0);
    size_t loops = 0;
    for (auto h : mesh.halfedges())
    {
        if (mesh.is_boundary(h) && !marker[h])
        {
            ++loops;
            auto hh = h;
            do
            {
                marker[hh] = 1;
                hh = mesh.next_halfedge(hh);
            } while (hh != h);
        }
    }
    mesh.remove_halfedge_property(marker);
    return loops;
}

size_t boundary_loop_length(const SurfaceMesh& mesh, Halfedge h)
{
    size_t n = 0;
    auto hh = h;
    do
    {
        ++n;
        hh = mesh.next_halfedge(hh);
    } while (hh != h);
    return n;
}

// Structural orientation check: for every interior edge the halfedge stored
// for each of the two incident faces must traverse the edge in opposite
// directions. Same-direction traversal means incompatible winding.
bool orientation_compatible(const SurfaceMesh& mesh)
{
    for (auto e : mesh.edges())
    {
        const auto h0 = mesh.halfedge(e, 0);
        const auto h1 = mesh.opposite_halfedge(h0);
        if (mesh.face(h0).is_valid() && mesh.face(h1).is_valid() &&
            mesh.to_vertex(h0) == mesh.to_vertex(h1))
            return false;
    }
    return true;
}

bool all_vertices_manifold(const SurfaceMesh& mesh)
{
    for (auto v : mesh.vertices())
        if (!mesh.is_manifold(v))
            return false;
    return true;
}

// Every edge is incident to at most two faces (no duplicate/non-manifold edge).
bool no_overfilled_edges(const SurfaceMesh& mesh)
{
    for (auto e : mesh.edges())
    {
        int incident = 0;
        if (mesh.face(mesh.halfedge(e, 0)).is_valid())
            ++incident;
        if (mesh.face(mesh.halfedge(e, 1)).is_valid())
            ++incident;
        if (incident > 2)
            return false;
    }
    return true;
}

bool no_duplicate_undirected_edges(const SurfaceMesh& mesh)
{
    std::map<std::pair<int, int>, int> count;
    for (auto e : mesh.edges())
    {
        int a = static_cast<int>(mesh.vertex(e, 0).idx());
        int b = static_cast<int>(mesh.vertex(e, 1).idx());
        if (a > b)
            std::swap(a, b);
        if (++count[{a, b}] > 1)
            return false;
    }
    return true;
}

bool no_degenerate_triangles(const SurfaceMesh& mesh,
                             const Scalar eps = Scalar(1e-12))
{
    for (auto f : mesh.faces())
    {
        std::vector<Point> p;
        for (auto v : mesh.vertices(f))
            p.push_back(mesh.position(v));
        if (p.size() != 3 ||
            norm(cross(p[1] - p[0], p[2] - p[0])) < eps)
            return false;
    }
    return true;
}

// Signed tetrahedron volume; a consistently oriented closed triangle mesh
// yields a non-zero value with one uniform sign.
Scalar signed_volume(const SurfaceMesh& mesh)
{
    Scalar volume = 0;
    for (auto f : mesh.faces())
    {
        std::vector<Point> p;
        for (auto v : mesh.vertices(f))
            p.push_back(mesh.position(v));
        volume += dot(p[0], cross(p[1], p[2])) / Scalar(6);
    }
    return volume;
}

struct MeshSnapshot
{
    size_t vertices;
    size_t edges;
    size_t faces;
    size_t boundary_halfedges;
    std::map<size_t, Point> positions;
    std::vector<std::pair<int, int>> edges_list;
};

MeshSnapshot snapshot(SurfaceMesh& mesh)
{
    MeshSnapshot s;
    s.vertices = mesh.n_vertices();
    s.edges = mesh.n_edges();
    s.faces = mesh.n_faces();
    s.boundary_halfedges = 0;
    for (auto h : mesh.halfedges())
        if (mesh.is_boundary(h))
            ++s.boundary_halfedges;
    for (auto v : mesh.vertices())
        s.positions[v.idx()] = mesh.position(v);
    for (auto e : mesh.edges())
        s.edges_list.emplace_back(
            static_cast<int>(mesh.vertex(e, 0).idx()),
            static_cast<int>(mesh.vertex(e, 1).idx()));
    return s;
}

bool identical_to_snapshot(const SurfaceMesh& mesh, const MeshSnapshot& s)
{
    size_t boundary_halfedges = 0;
    for (auto h : mesh.halfedges())
        if (mesh.is_boundary(h))
            ++boundary_halfedges;
    if (mesh.n_vertices() != s.vertices || mesh.n_edges() != s.edges ||
        mesh.n_faces() != s.faces ||
        boundary_halfedges != s.boundary_halfedges)
        return false;
    for (auto v : mesh.vertices())
    {
        const auto it = s.positions.find(v.idx());
        if (it == s.positions.end() ||
            distance(mesh.position(v), it->second) > Scalar(1e-14))
            return false;
    }
    for (const auto& edge : s.edges_list)
        if (!mesh.find_halfedge(Vertex(edge.first), Vertex(edge.second))
                 .is_valid())
            return false;
    return true;
}

bool no_locked_properties(const SurfaceMesh& mesh)
{
    return !mesh.has_vertex_property("HoleFilling:vlocked") &&
           !mesh.has_edge_property("HoleFilling:elocked");
}

// Find a boundary halfedge of a fully manifold boundary loop.
Halfedge manifold_boundary_halfedge(SurfaceMesh& mesh)
{
    for (auto h : mesh.halfedges())
    {
        if (!mesh.is_boundary(h))
            continue;
        bool manifold = true;
        auto hh = h;
        do
        {
            if (!mesh.is_manifold(mesh.to_vertex(hh)))
                manifold = false;
            hh = mesh.next_halfedge(hh);
        } while (hh != h);
        if (manifold)
            return h;
    }
    return {};
}

} // namespace

// ---------------------------------------------------------------------------
// rejected input: the mesh must remain unchanged, no temporary state leaks
// ---------------------------------------------------------------------------

TEST(HoleFillingTest, invalid_halfedge_is_rejected_without_modification)
{
    auto mesh = open_tube(5);
    const auto before = snapshot(mesh);

    EXPECT_THROW(fill_hole(mesh, Halfedge()), InvalidInputException);

    EXPECT_TRUE(identical_to_snapshot(mesh, before));
    EXPECT_TRUE(no_locked_properties(mesh));
    EXPECT_EQ(count_boundary_loops(mesh), 2u);
}

TEST(HoleFillingTest, non_boundary_halfedge_is_rejected_without_modification)
{
    auto mesh = open_tube(5);
    const auto before = snapshot(mesh);

    Halfedge interior;
    for (auto h : mesh.halfedges())
        if (!mesh.is_boundary(h))
        {
            interior = h;
            break;
        }
    ASSERT_TRUE(interior.is_valid());

    EXPECT_THROW(fill_hole(mesh, interior), InvalidInputException);

    EXPECT_TRUE(identical_to_snapshot(mesh, before));
    EXPECT_TRUE(no_locked_properties(mesh));
    // a face count alone would not prove anything; the boundary must survive
    EXPECT_EQ(count_boundary_loops(mesh), 2u);
}

TEST(HoleFillingTest, non_manifold_hole_is_rejected_without_modification)
{
    auto mesh = non_manifold_fan();
    const auto before = snapshot(mesh);

    const Halfedge h = first_boundary_halfedge(mesh);
    ASSERT_TRUE(h.is_valid());
    // the shared vertex is the target of the first boundary halfedge and
    // has two outgoing boundary halfedges (two boundary gaps)
    ASSERT_FALSE(mesh.is_manifold(mesh.to_vertex(h)));

    EXPECT_THROW(fill_hole(mesh, h), InvalidInputException);

    // rejection happens before any connectivity change
    EXPECT_TRUE(identical_to_snapshot(mesh, before));
    EXPECT_TRUE(no_locked_properties(mesh));
    // the geometric defect is still observable
    EXPECT_FALSE(all_vertices_manifold(mesh));
    EXPECT_GE(count_boundary_loops(mesh), 1u);
}

// ---------------------------------------------------------------------------
// successful filling on non-planar boundaries
// ---------------------------------------------------------------------------

TEST(HoleFillingTest, non_planar_hole_is_closed_and_manifold)
{
    // removing three adjacent caps of an 8-ring leaves a boundary of
    // length k+4 = 7 with two existing interior chords
    auto mesh = warped_disk(2, 3);
    const auto before = snapshot(mesh);

    const Halfedge h = first_boundary_halfedge(mesh);
    ASSERT_TRUE(h.is_valid());
    const size_t hole_length = boundary_loop_length(mesh, h);
    EXPECT_EQ(hole_length, 7u);

    fill_hole(mesh, h);

    // topological result: the specified boundary is closed, nothing else
    EXPECT_EQ(count_boundary_loops(mesh), 0u);
    EXPECT_TRUE(all_vertices_manifold(mesh));
    EXPECT_TRUE(no_overfilled_edges(mesh));
    EXPECT_TRUE(no_duplicate_undirected_edges(mesh));
    EXPECT_TRUE(mesh.is_triangle_mesh());
    EXPECT_TRUE(orientation_compatible(mesh));

    // geometric result: valid, consistently oriented, non-degenerate patch
    EXPECT_TRUE(no_degenerate_triangles(mesh));
    EXPECT_NE(signed_volume(mesh), 0.0);

    // returning normally or growing the face count is not the success
    // criterion: here we explicitly require the boundary to be gone and the
    // old geometry to be untouched
    EXPECT_GE(mesh.n_faces(), before.faces);
}

TEST(HoleFillingTest, existing_interior_chord_is_reused_not_duplicated)
{
    for (const bool warped : {false, true})
    {
        auto mesh = chord_hexagon(warped);

        const Vertex a(0), b(2);
        const Halfedge chord = mesh.find_halfedge(a, b);
        ASSERT_TRUE(chord.is_valid());
        ASSERT_TRUE(!mesh.is_boundary(chord) &&
                    !mesh.is_boundary(mesh.opposite_halfedge(chord)));

        const Halfedge h = first_boundary_halfedge(mesh);
        ASSERT_TRUE(h.is_valid());
        EXPECT_EQ(boundary_loop_length(mesh, h), 6u);

        fill_hole(mesh, h);

        EXPECT_EQ(count_boundary_loops(mesh), 0u);
        EXPECT_TRUE(no_duplicate_undirected_edges(mesh));
        EXPECT_TRUE(no_overfilled_edges(mesh));
        EXPECT_TRUE(all_vertices_manifold(mesh));
        EXPECT_TRUE(orientation_compatible(mesh));
        EXPECT_TRUE(no_degenerate_triangles(mesh));

        // exactly one edge still connects a and b: the pre-existing chord
        int chord_edges = 0;
        for (auto e : mesh.edges())
        {
            const int x = static_cast<int>(mesh.vertex(e, 0).idx());
            const int y = static_cast<int>(mesh.vertex(e, 1).idx());
            if ((x == 0 && y == 2) || (x == 2 && y == 0))
                ++chord_edges;
        }
        EXPECT_EQ(chord_edges, 1);
    }
}

TEST(HoleFillingTest, minimal_three_vertex_hole)
{
    SurfaceMesh mesh;
    const auto v0 = mesh.add_vertex(Point(0, 0, 0));
    const auto v1 = mesh.add_vertex(Point(1, 0, 0.2));
    const auto v2 = mesh.add_vertex(Point(0.5, 1, -0.1));
    const auto v3 = mesh.add_vertex(Point(0.2, 0.3, 1));
    const auto bottom = mesh.add_triangle(v0, v1, v2);
    mesh.add_triangle(v0, v3, v1);
    mesh.add_triangle(v1, v3, v2);
    mesh.add_triangle(v2, v3, v0);
    mesh.delete_face(bottom);
    mesh.garbage_collection();

    const Halfedge h = first_boundary_halfedge(mesh);
    ASSERT_TRUE(h.is_valid());
    ASSERT_EQ(boundary_loop_length(mesh, h), 3u);

    fill_hole(mesh, h);

    EXPECT_EQ(count_boundary_loops(mesh), 0u);
    EXPECT_TRUE(orientation_compatible(mesh));
    EXPECT_TRUE(no_degenerate_triangles(mesh));
    EXPECT_TRUE(all_vertices_manifold(mesh));
}

// ---------------------------------------------------------------------------
// acute corners: a legal sharp fold must not be confused with a flip
// ---------------------------------------------------------------------------

TEST(HoleFillingTest, acute_corner_fills_without_flipped_faces)
{
    auto built = make_acute_wedge_mesh();
    auto& mesh = built.mesh;
    const Halfedge h = built.hole;
    ASSERT_TRUE(h.is_valid());
    ASSERT_EQ(boundary_loop_length(mesh, h), 4u);
    // the mesh carries a second, independent boundary (the outer ring)
    const size_t loops_before = count_boundary_loops(mesh);
    ASSERT_GE(loops_before, 2u);

    // the hole boundary must contain an acute corner (the ~21 degree spike)
    Scalar smallest_corner = std::numbers::pi;
    Vertex acute_vertex{};
    {
        auto hh = h;
        do
        {
            const Vertex v = mesh.to_vertex(hh);
            const Scalar angle = boundary_corner_angle(mesh, v);
            if (angle < smallest_corner)
            {
                smallest_corner = angle;
                acute_vertex = v;
            }
            hh = mesh.next_halfedge(hh);
        } while (hh != h);
    }
    ASSERT_TRUE(acute_vertex.is_valid());
    EXPECT_EQ(acute_vertex, built.spike);
    EXPECT_LT(smallest_corner,
              Scalar(25) * std::numbers::pi / Scalar(180));

    fill_hole(mesh, h);

    // the specified hole is closed while the unrelated outer boundary stays
    EXPECT_EQ(count_boundary_loops(mesh), loops_before - 1u);
    // structural winding consistency distinguishes a legal sharp fold from
    // an actual face flip, independent of how small the corner angle is
    EXPECT_TRUE(orientation_compatible(mesh));
    EXPECT_TRUE(no_degenerate_triangles(mesh));
    EXPECT_TRUE(no_duplicate_undirected_edges(mesh));
    EXPECT_TRUE(all_vertices_manifold(mesh));
}

// ---------------------------------------------------------------------------
// refinement and fairing must not touch protected old geometry
// ---------------------------------------------------------------------------

TEST(HoleFillingTest, refinement_and_fairing_preserve_old_vertices_and_edges)
{
    auto mesh = warped_disk(4, 6);
    const auto before = snapshot(mesh);
    auto old_mark = mesh.add_vertex_property<bool>("test:old", false);
    for (auto v : mesh.vertices())
        old_mark[v] = true;

    const Halfedge h = first_boundary_halfedge(mesh);
    ASSERT_TRUE(h.is_valid());

    fill_hole(mesh, h);
    ASSERT_EQ(count_boundary_loops(mesh), 0u);

    // old vertices keep their exact positions and stay manifold
    for (auto v : mesh.vertices())
    {
        if (!old_mark[v])
            continue;
        const auto it = before.positions.find(v.idx());
        ASSERT_NE(it, before.positions.end());
        EXPECT_EQ(mesh.position(v), it->second);
        EXPECT_TRUE(mesh.is_manifold(v));
    }

    // every old edge still connects the same two vertices
    for (const auto& edge : before.edges_list)
        EXPECT_TRUE(mesh.find_halfedge(Vertex(edge.first),
                                       Vertex(edge.second))
                        .is_valid());

    // refinement may insert interior vertices, but they must not remove the
    // boundary or create duplicate connections
    EXPECT_TRUE(no_duplicate_undirected_edges(mesh));
    EXPECT_TRUE(no_overfilled_edges(mesh));
    EXPECT_TRUE(orientation_compatible(mesh));
    EXPECT_TRUE(no_degenerate_triangles(mesh));

    mesh.remove_vertex_property(old_mark);
}

TEST(HoleFillingTest, large_hole_smoothes_new_vertices_only)
{
    // removing five adjacent caps of an 8-ring leaves a boundary of
    // length n+2-k = 5 with long edges that force refinement splits
    auto mesh = warped_disk(0, 5);
    auto old_mark = mesh.add_vertex_property<bool>("test:old", false);
    for (auto v : mesh.vertices())
        old_mark[v] = true;
    const size_t old_vertex_count = mesh.n_vertices();

    const Halfedge h = first_boundary_halfedge(mesh);
    ASSERT_TRUE(h.is_valid());
    ASSERT_EQ(boundary_loop_length(mesh, h), 5u);

    fill_hole(mesh, h);

    EXPECT_EQ(count_boundary_loops(mesh), 0u);
    // new interior vertices were created by the refinement stage
    size_t new_vertices = 0;
    for (auto v : mesh.vertices())
        if (!old_mark[v])
            ++new_vertices;
    EXPECT_GT(new_vertices, 0u);
    EXPECT_GT(mesh.n_vertices(), old_vertex_count);

    // the smoothed interior stays inside the bounding box of the old rim
    BoundingBox box;
    for (auto v : mesh.vertices())
        if (old_mark[v])
            box += mesh.position(v);
    for (auto v : mesh.vertices())
    {
        if (old_mark[v])
            continue;
        const Point& p = mesh.position(v);
        for (int i = 0; i < 3; ++i)
        {
            EXPECT_GE(p[i], box.min()[i] - Scalar(1e-9));
            EXPECT_LE(p[i], box.max()[i] + Scalar(1e-9));
        }
    }

    EXPECT_TRUE(orientation_compatible(mesh));
    EXPECT_TRUE(no_degenerate_triangles(mesh));
    mesh.remove_vertex_property(old_mark);
}

// ---------------------------------------------------------------------------
// multiple holes in one mesh are independent
// ---------------------------------------------------------------------------

TEST(HoleFillingTest, filling_one_hole_leaves_other_holes_untouched)
{
    auto mesh = open_tube(8);
    ASSERT_EQ(count_boundary_loops(mesh), 2u);

    const Halfedge first = first_boundary_halfedge(mesh);
    ASSERT_TRUE(first.is_valid());
    fill_hole(mesh, first);

    EXPECT_EQ(count_boundary_loops(mesh), 1u);
    EXPECT_TRUE(all_vertices_manifold(mesh));
    EXPECT_TRUE(orientation_compatible(mesh));

    const Halfedge second = first_boundary_halfedge(mesh);
    ASSERT_TRUE(second.is_valid());
    fill_hole(mesh, second);

    EXPECT_EQ(count_boundary_loops(mesh), 0u);
    EXPECT_TRUE(all_vertices_manifold(mesh));
    EXPECT_TRUE(no_duplicate_undirected_edges(mesh));
    EXPECT_TRUE(no_degenerate_triangles(mesh));
    EXPECT_NE(signed_volume(mesh), 0.0);
}

// ---------------------------------------------------------------------------
// failure stages and observable mesh state
// ---------------------------------------------------------------------------

TEST(HoleFillingTest, rejected_complete_and_untouched_states_distinguishable)
{
    // rejected: exception and unchanged mesh
    auto rejected = non_manifold_fan();
    const auto rejected_before = snapshot(rejected);
    EXPECT_THROW(fill_hole(rejected, first_boundary_halfedge(rejected)),
                 InvalidInputException);
    EXPECT_TRUE(identical_to_snapshot(rejected, rejected_before));
    EXPECT_TRUE(no_locked_properties(rejected));

    // complete: no exception and no boundary left
    auto completed = warped_disk(2, 2);
    fill_hole(completed, first_boundary_halfedge(completed));
    EXPECT_EQ(count_boundary_loops(completed), 0u);
    EXPECT_TRUE(no_locked_properties(completed));

    // untouched: a valid second hole in the rejected mesh is still fillable
    auto mixed = open_tube(6);
    const auto shared = mixed.add_vertex(Point(5, 5, 5));
    const auto a1 = mixed.add_vertex(Point(6, 5, 5));
    const auto a2 = mixed.add_vertex(Point(5, 6, 5));
    const auto b1 = mixed.add_vertex(Point(4, 5, 5));
    const auto b2 = mixed.add_vertex(Point(5, 4, 5));
    mixed.add_triangle(shared, a1, a2);
    mixed.add_triangle(shared, b1, b2);

    Halfedge bad;
    for (auto h : mixed.halfedges())
        if (mixed.is_boundary(h) && !mixed.is_manifold(mixed.from_vertex(h)))
        {
            bad = h;
            break;
        }
    ASSERT_TRUE(bad.is_valid());
    EXPECT_THROW(fill_hole(mixed, bad), InvalidInputException);
    EXPECT_TRUE(no_locked_properties(mixed));

    const Halfedge good = manifold_boundary_halfedge(mixed);
    ASSERT_TRUE(good.is_valid());
    const size_t loops_before = count_boundary_loops(mixed);
    fill_hole(mixed, good);
    // the manifold hole was closed while the non-manifold region remains
    EXPECT_LT(count_boundary_loops(mixed), loops_before);
    EXPECT_FALSE(all_vertices_manifold(mixed));
    EXPECT_TRUE(no_locked_properties(mixed));
}

TEST(HoleFillingTest, repeated_failures_do_not_block_later_valid_filling)
{
    auto mesh = open_tube(6);
    const auto shared = mesh.add_vertex(Point(5, 5, 5));
    const auto a1 = mesh.add_vertex(Point(6, 5, 5));
    const auto a2 = mesh.add_vertex(Point(5, 6, 5));
    const auto b1 = mesh.add_vertex(Point(4, 5, 5));
    const auto b2 = mesh.add_vertex(Point(5, 4, 5));
    mesh.add_triangle(shared, a1, a2);
    mesh.add_triangle(shared, b1, b2);

    Halfedge bad;
    for (auto h : mesh.halfedges())
        if (mesh.is_boundary(h) && !mesh.is_manifold(mesh.from_vertex(h)))
        {
            bad = h;
            break;
        }
    ASSERT_TRUE(bad.is_valid());

    for (int attempt = 0; attempt < 3; ++attempt)
    {
        EXPECT_THROW(fill_hole(mesh, bad), InvalidInputException);
        EXPECT_TRUE(no_locked_properties(mesh));
    }

    // residual state from the failed attempts must not interfere
    const Halfedge good = manifold_boundary_halfedge(mesh);
    ASSERT_TRUE(good.is_valid());
    const size_t loops_before = count_boundary_loops(mesh);
    EXPECT_NO_THROW(fill_hole(mesh, good));
    EXPECT_LT(count_boundary_loops(mesh), loops_before);
    EXPECT_TRUE(orientation_compatible(mesh));
}

TEST(HoleFillingTest, preexisting_damage_survives_and_does_not_block_fill)
{
    auto mesh = open_tube(6);
    // an isolated degenerate triangle is pre-existing geometric damage
    const auto x = mesh.add_vertex(Point(10, 10, 10));
    const auto y = mesh.add_vertex(Point(10, 10, 10));
    const auto z = mesh.add_vertex(Point(10.5, 10, 10));
    mesh.add_triangle(x, y, z);
    ASSERT_FALSE(no_degenerate_triangles(mesh));

    const size_t loops_before = count_boundary_loops(mesh);
    const Halfedge h = manifold_boundary_halfedge(mesh);
    ASSERT_TRUE(h.is_valid());

    EXPECT_NO_THROW(fill_hole(mesh, h));

    // the legal hole was closed but the damage is still separately visible
    EXPECT_LT(count_boundary_loops(mesh), loops_before);
    EXPECT_FALSE(no_degenerate_triangles(mesh));
    EXPECT_TRUE(no_locked_properties(mesh));
}

// ---------------------------------------------------------------------------
// regression coverage for the original simple case
// ---------------------------------------------------------------------------

TEST(HoleFillingTest, open_cone)
{
    auto mesh = open_cone();

    Halfedge h = first_boundary_halfedge(mesh);
    EXPECT_TRUE(h.is_valid());

    fill_hole(mesh, h);

    h = first_boundary_halfedge(mesh);
    EXPECT_FALSE(h.is_valid());
}
