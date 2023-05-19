use super::ChunkShape;
use crate::voxel::{storage::ChunkMap, Voxel};
use bevy::{math::Vec3Swizzles, prelude::*};
use itertools::{iproduct, Itertools};

pub const SIMULATION_STEP: f32 = 0.05;

#[derive(Bundle)]
pub struct ColliderBundle {
    pub acceleration: Acceleration,
    pub velocity: Velocity,
    pub transform: Transform,
    pub collider: Collider3d,
}

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, SystemSet)]
/// Systems which apply physics to objects.
pub enum PhysicsSet {
    /// Apply acceleration to entities with both [Acceleration] and [Velocity].
    Acceleration,
    /// Apply [Velocity] to entities with both [Velocity] and [Transform], considering collisions for those with [VoxelCollider3d].
    Velocity,
    /// Apply [Drag] to update [Velocity].
    Drag,
}

pub struct PhysicsPlugin;

impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(FixedTime::new_from_secs(SIMULATION_STEP))
            .configure_sets(
                (
                    PhysicsSet::Acceleration,
                    PhysicsSet::Velocity.ambiguous_with(PhysicsSet::Velocity),
                    PhysicsSet::Drag,
                )
                    .in_base_set(CoreSet::FixedUpdate)
                    .chain(),
            )
            .add_system(apply_acceleration.in_set(PhysicsSet::Acceleration))
            .add_systems(
                (
                    apply_velocity_for_colliders,
                    apply_velocity_for_non_colliders,
                )
                    .in_set(PhysicsSet::Velocity),
            )
            .add_system(apply_drag.in_set(PhysicsSet::Drag));
    }
}

fn apply_acceleration(mut objects: Query<(&Acceleration, &mut Velocity)>) {
    for (a, mut v) in &mut objects {
        **v += **a * SIMULATION_STEP;
    }
}

fn apply_velocity_for_non_colliders(
    mut objects: Query<(&Velocity, &mut Transform), Without<Collider3d>>,
) {
    for (v, mut t) in &mut objects {
        t.translation += **v * SIMULATION_STEP;
    }
}

fn apply_velocity_for_colliders(
    mut objects: Query<(&Collider3d, &mut Velocity, &mut Transform)>,
    voxels: Res<ChunkMap<Voxel, ChunkShape>>,
) {
    for (c, mut v, mut t) in &mut objects {
        let aabb_half_extents = c.aabb_half_extents();
        let mut time_remaining = SIMULATION_STEP;
        let voxel_collider = Collider3d::AxisAlignedCube { radius: 0.5 };
        for _ in 0..3 {
            // at most 3 iterations, one for each axis
            let min = Vec3::min(
                t.translation - aabb_half_extents,
                t.translation - aabb_half_extents + **v * time_remaining,
            )
            .floor()
            .as_ivec3();

            let max = Vec3::max(
                t.translation + aabb_half_extents,
                t.translation + aabb_half_extents + **v * time_remaining,
            )
            .floor()
            .as_ivec3();

            println!();
            println!(
                "{} @ {} @ {}: {} -> {}",
                t.translation, aabb_half_extents, **v, min, max
            );
            if let Some(collision) = iproduct!(min.x..=max.x, min.y..=max.y, min.z..=max.z)
                .map(|(x, y, z)| IVec3::new(x, y, z))
                .filter(|voxel| voxels.voxel_at(*voxel).is_some_and(Voxel::collidable))
                .inspect(|v| println!("candidate voxel: {}", v))
                .flat_map(|voxel| {
                    c.swept_collision(**v, t.translation, voxel_collider, voxel.as_vec3() + 0.5)
                })
                .inspect(|c| println!("candidate collision time: {}", c.time))
                .min_by(|a, b| a.time.total_cmp(&b.time))
                // discard collision if it happens after the simulation period
                .filter(|c| c.time <= time_remaining)
            {
                t.translation += **v * (collision.time - 1e-6).max(0.);
                let p_v = **v;
                // cancel velocity in the normal direction
                **v -= Vec3::dot(p_v, collision.normal) * collision.normal;
                time_remaining -= collision.time;
            } else {
                t.translation += time_remaining * **v;
                break; // no more steps needed
            }
        }
        // println!("{} += {}", t.translation, displacement);
    }
}

fn apply_drag(mut objects: Query<(&Drag, &mut Velocity)>) {
    for (d, mut v) in &mut objects {
        **v *= **d;
    }
}

#[derive(Clone, Copy, Component, Default, Debug, Deref, DerefMut)]
pub struct Acceleration(pub Vec3);

#[derive(Clone, Copy, Component, Default, Debug, Deref, DerefMut)]
pub struct Velocity(pub Vec3);

#[derive(Clone, Copy, Component, Debug, Deref, DerefMut)]
pub struct Drag(pub f32);

/// Information about a collision in 3D.
struct Collision3d {
    time: f32,
    normal: Vec3,
}

#[cfg(test)]
impl std::fmt::Debug for Collision3d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut buffer = ryu::Buffer::new();
        let time = buffer.format(self.time).to_owned();
        let x = buffer.format(self.normal.x).to_owned();
        let y = buffer.format(self.normal.y).to_owned();
        let z = buffer.format(self.normal.z).to_owned();
        // truncate the values to make snapshots cross-platform (since floating point errors are terrible)
        write!(
            f,
            "Collision3d {{\n    time: {},\n    normal: Vec3(\n        {},\n        {},\n        {},\n    ),\n}}",            
            time,
            x,
            y,
            z
        )
    }
}

struct Collision2d {
    time: f32,
    normal: Vec2,
}

#[cfg(test)]
impl std::fmt::Debug for Collision2d {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut buffer = ryu::Buffer::new();
        let time = buffer.format(self.time).to_owned();
        let x = buffer.format(self.normal.x).to_owned();
        let y = buffer.format(self.normal.y).to_owned();
        // truncate the values to make snapshots cross-platform (since floating point errors are terrible)
        write!(
            f,
            "Collision2d {{\n    time: {},\n    normal: Vec2(\n        {},\n        {},\n    ),\n}}",            
            time,
            x,
            y,
        )
    }
}

#[derive(Component, Debug, Copy, Clone)]
/// A physics object which collides with voxels (todo: and other colliders?).
/// Collisions with terrain are resolved by clamping and adjusting velocity.
pub enum Collider3d {
    AxisAlignedCylinder { radius: f32, half_height: f32 },
    AxisAlignedCube { radius: f32 },
}

impl Collider3d {
    fn aabb_half_extents(&self) -> Vec3 {
        match *self {
            Collider3d::AxisAlignedCylinder {
                radius,
                half_height,
            } => Vec3::new(radius, half_height, radius),
            Collider3d::AxisAlignedCube { radius } => Vec3::splat(radius),
        }
    }

    fn swept_collision(
        &self,
        self_velocity: Vec3,
        self_center: Vec3,
        other: Self,
        other_center: Vec3,
    ) -> Option<Collision3d> {
        match (*self, other) {
            (
                Collider3d::AxisAlignedCylinder {
                    radius: cyl_radius,
                    half_height: cyl_half_height,
                },
                Collider3d::AxisAlignedCube {
                    radius: cube_radius,
                },
            ) => swept_aacyl_aacube_collision(
                self_velocity,
                self_center,
                cyl_radius,
                cyl_half_height,
                other_center,
                cube_radius,
            ),
            _ => todo!("unimplemented swept collision test"),
        }
    }
}

/// Returns the time of intersection between a ray and a line segment, if one exists.
/// The ray points in the direction of `v` and time is scaled according to the magnitude of `v`.
///
/// Note that no collision will be found if there is more than one intersection (i.e. the ray and segment are
/// parallel). This is desirable for our use-case because it means we will find the correct normal.
fn ray_segment_collision(o: Vec2, v: Vec2, a: Vec2, b: Vec2) -> Option<Collision2d> {
    // https://rootllama.wordpress.com/2014/06/20/ray-line-segment-intersection-test-in-2d/
    let v1 = o - a;
    let v2 = b - a;
    let v3 = Vec2::new(-v.y, v.x);
    let t1 = v2.perp_dot(v1) / (v2.dot(v3));
    let t2 = v1.dot(v3) / v2.dot(v3);
    (t1 >= 0. && 0. <= t2 && t2 <= 1.).then_some(Collision2d {
        time: t1,
        normal: (-v).normalize(),
    })
}

fn swept_circle_segment_collision(
    r: f32,
    c: Vec2,
    v: Vec2,
    a: Vec2,
    b: Vec2,
) -> Option<Collision2d> {
    // https://stackoverflow.com/a/7060627

    // first, check if the center of the circle collides with the segment (obvious case)
    ray_segment_collision(c + r * v.normalize(), v, a, b).or_else(|| {
        // failing the obvious case, we need to check whether the center of the circle comes within `r` from the
        // endpoints of the segment.
        [a, b]
            .into_iter()
            .filter_map(|x| {
                let p = (x - c).project_onto(v);
                if p.dot(v) >= 0.0 {
                    ray_segment_collision(c, v, x, p + c)
                        .and_then(|d| (d.time <= r).then_some(p.length()))
                        .map(|time| Collision2d {
                            time,
                            normal: (p - x).normalize(),
                        })
                } else {
                    None
                }
            })
            .min_by(|a, b| a.time.total_cmp(&b.time))
    })
}

/// utility function to compute whether a circle and square touch or intersect in 2D
fn circle_square_intersect(
    circle_center: Vec2,
    circle_radius: f32,
    square_center: Vec2,
    square_radius: f32,
) -> bool {
    // see: https://stackoverflow.com/questions/401847/circle-rectangle-collision-detection-intersection
    let cd = (circle_center - square_center).abs();
    if cd.max_element() > square_radius + circle_radius {
        // circle is too far to possibly intersect
        false
    } else if cd.min_element() <= square_radius {
        // circle is close enough to guarantee an intersection
        true
    } else {
        // check if the circle clips the corner of the square
        let corner_dist_sq = (cd - square_radius).length_squared();
        corner_dist_sq <= circle_radius.powi(2)
    }
}

/// Computes the first collision between a moving axis-aligned cylinder and stationary cube, if one exists.
///
/// Assumes the cylinder and cube are initially not overlapping (but may be touching).
/// Will not report a collision unless the cylinder and cube overlap (not just touch).
fn swept_aacyl_aacube_collision(
    cyl_velocity: Vec3,
    cyl_center: Vec3,
    cyl_radius: f32,
    cyl_half_height: f32,
    cube_center: Vec3,
    cube_radius: f32,
) -> Option<Collision3d> {
    assert!(cyl_radius > 0., "cylinder radius must be positive");
    assert!(cube_radius > 0., "cube radius must be positive");

    // We can reduce the problem to 2D by projecting the cylinder and cube onto the xz axis (since both are
    // aligned on the y-axis).
    // Then, we sweep the circle accoridng to the xz component of the cylinder's velocity, and determine whether
    // the circle collides with any of the four edges of the cube. If so, we can determine whether the cylinder
    // and cube overlapped on the y axis at the time of collision.

    let (t_start, t_end) = if cyl_velocity.y == 0. {
        if (cube_center.y - cyl_center.y).abs() < cyl_half_height + cube_radius {
            (0., f32::INFINITY)
        } else {
            return None;
        }
    } else {
        // displacement from bottom of cube to top of cylinder
        let d1 = (cube_center.y - cube_radius) - (cyl_center.y + cyl_half_height);
        // displacement from top of cube to bottom of cylinder
        let d2 = (cube_center.y + cube_radius) - (cyl_center.y - cyl_half_height);
        let (t1, t2) = (d1 / cyl_velocity.y, d2 / cyl_velocity.y);
        let min = t1.min(t2);
        let max = t1.max(t2);
        if max < 0.0 {
            (0.0, f32::INFINITY)
        } else {
            (min.max(0.0), max)
        }
    };

    // If the cylinder and cube collide at time t, we know t_start <= t <= t_end.
    // We just need to sweep the circle, find any collisions with the segments of the square,
    // and determine if the collisions happen in that interval.

    let circle_velocity = cyl_velocity.xz();
    let circle_center = cyl_center.xz();
    let square_center = cube_center.xz();

    if circle_velocity == Vec2::ZERO {
        // we assumed initially not overlapping, so the only possible intersections are with the top/bottom faces
        if circle_square_intersect(circle_center, cyl_radius, square_center, cube_radius) {
            if cyl_velocity.y == 0.0 {
                // stationary and initially non-colliding means we are still not colliding
                None
            } else {
                // we collide at t_start with either the top or bottom face
                let normal = if cyl_velocity.y.is_sign_positive() {
                    Vec3::NEG_Y
                } else {
                    Vec3::Y
                };
                Some(Collision3d {
                    time: t_start,
                    normal,
                })
            }
        } else {
            // not moving in xz plane and not overlapping means no collision is possible
            None
        }
    } else {
        // need to perform swept circle-segment tests for the four edges
        let corners = [
            Vec2::new(cyl_radius, -cyl_radius),  // bottom right
            Vec2::new(cyl_radius, cyl_radius),   // top right
            Vec2::new(-cyl_radius, cyl_radius),  // top left
            Vec2::new(-cyl_radius, -cyl_radius), // bottom left
        ]
        .map(|offset| square_center + offset);

        corners
            .into_iter()
            .cycle()
            .tuple_windows()
            .take(4)
            .filter_map(|(a, b)| {
                swept_circle_segment_collision(cyl_radius, circle_center, circle_velocity, a, b)
            })
            .min_by(|c1, c2| c1.time.total_cmp(&c2.time))
            .filter(|c| t_start <= c.time && c.time <= t_end)
            .map(|c| Collision3d {
                time: c.time,
                normal: c.normal.extend(0.).xzy(),
            })
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use insta::assert_debug_snapshot;

    #[test]
    fn test_ray_segment_intersect() {
        assert_debug_snapshot!(ray_segment_collision(Vec2::Y * 0.5, Vec2::X, Vec2::X, Vec2::ONE), @r###"
        Some(
            Collision2d {
                time: 1.0,
                normal: Vec2(
                    -1.0,
                    -0.0,
                ),
            },
        )
        "###);
        assert_debug_snapshot!(
            ray_segment_collision(Vec2::ZERO, Vec2::ONE, Vec2::X, Vec2::Y),
            @r###"
        Some(
            Collision2d {
                time: 0.5,
                normal: Vec2(
                    -0.70710677,
                    -0.70710677,
                ),
            },
        )
        "###
        );
        assert!(ray_segment_collision(Vec2::ZERO, Vec2::new(-1., 1.), Vec2::X, Vec2::Y).is_none());
        assert!(ray_segment_collision(Vec2::ONE, Vec2::ONE, Vec2::X, Vec2::Y).is_none());
        assert_debug_snapshot!(
            ray_segment_collision(Vec2::ONE, Vec2::NEG_ONE, Vec2::X, Vec2::Y), @r###"
        Some(
            Collision2d {
                time: 0.5,
                normal: Vec2(
                    0.70710677,
                    0.70710677,
                ),
            },
        )
        "###
        );
        assert!(ray_segment_collision(Vec2::ZERO, Vec2::X, Vec2::X, Vec2::X * 2.).is_none());
        assert_debug_snapshot!(
            ray_segment_collision(Vec2::ZERO, Vec2::ONE * 0.25, Vec2::X, Vec2::Y),
        @r###"
        Some(
            Collision2d {
                time: 2.0,
                normal: Vec2(
                    -0.70710677,
                    -0.70710677,
                ),
            },
        )
        "###
        );
    }

    #[test]
    fn test_swept_circle_segment_collision() {
        // assert_f32_near!(
        //     swept_circle_segment_collision(0.5, Vec2::ZERO, Vec2::X, Vec2::ONE, Vec2::X).unwrap(),
        //     0.5
        // );
        assert!(
            swept_circle_segment_collision(0.5, Vec2::ZERO, Vec2::NEG_X, Vec2::ONE, Vec2::X)
                .is_none()
        );
        // todo add more test cases
    }

    #[test]
    fn test_swept_aacyl_aacube_collision() {
        assert_debug_snapshot!(swept_aacyl_aacube_collision(Vec3::X, Vec3::ZERO, 0.5, 0.5, Vec3::X * 5.0, 0.5), @r###"
        Some(
            Collision3d {
                time: 4.0,
                normal: Vec3(
                    -1.0,
                    0.0,
                    -0.0,
                ),
            },
        )
        "###);

        assert_debug_snapshot!(swept_aacyl_aacube_collision(Vec3::Y, Vec3::ZERO, 0.5, 0.5, Vec3::Y * 2.0, 1.0), @r###"
        Some(
            Collision3d {
                time: 0.5,
                normal: Vec3(
                    0.0,
                    -1.0,
                    0.0,
                ),
            },
        )
        "###);

        assert_debug_snapshot!(swept_aacyl_aacube_collision(Vec3::NEG_Y, Vec3::ZERO, 1.0, 0.5, Vec3::NEG_Y * 2.0, 1.0), @r###"
        Some(
            Collision3d {
                time: 0.5,
                normal: Vec3(
                    0.0,
                    1.0,
                    0.0,
                ),
            },
        )
        "###);

        assert_debug_snapshot!(swept_aacyl_aacube_collision(Vec3::new(1.0, 0.0, 1.0), Vec3::ZERO, 0.5, 0.5, Vec3::new(1.0, 0.0, 1.0), 0.5), @r###"
        Some(
            Collision3d {
                time: 0.14644662,
                normal: Vec3(
                    -0.70710677,
                    0.0,
                    -0.70710677,
                ),
            },
        )
        "###);

        assert_debug_snapshot!(swept_aacyl_aacube_collision(Vec3::new(0.0, -0.01, 0.0), Vec3::new(-2.1, 129.5, -10.25), 0.4, 0.9, Vec3::new(-2.5, 127.5, -10.5), 0.5), @r###"
        Some(
            Collision3d {
                time: 60.00061,
                normal: Vec3(
                    0.0,
                    1.0,
                    0.0,
                ),
            },
        )
        "###);

        assert_debug_snapshot!(
            swept_aacyl_aacube_collision(Vec3::new(0., -0.2, 0.), Vec3::new(-6., 128.91, -19.2), 0.4, 0.9, Vec3::new(-6.5, 127.5, -19.5), 0.5),
            @r###"
        Some(
            Collision3d {
                time: 0.050048828,
                normal: Vec3(
                    0.0,
                    1.0,
                    0.0,
                ),
            },
        )
        "###
        );
    }
}
