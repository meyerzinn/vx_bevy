use super::ChunkShape;
use crate::voxel::{storage::ChunkMap, Voxel};
use bevy::{
    ecs::schedule::BaseSystemSet,
    prelude::*,
    time::{
        common_conditions::{on_fixed_timer, on_timer},
        fixed_timestep,
    },
};
use itertools::{iproduct, Itertools};
use std::{ops::Neg, time::Duration};

pub const SIMULATION_STEP: f32 = 0.05;

#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, SystemSet)]
pub enum PhysicsSet {
    Acceleration,
    Velocity,
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
                    .chain()
                    .in_base_set(CoreSet::FixedUpdate),
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
    mut objects: Query<(&Velocity, &mut Transform), Without<TerrainCollider>>,
) {
    for (v, mut t) in &mut objects {
        t.translation += **v * SIMULATION_STEP;
    }
}

fn apply_velocity_for_colliders(
    mut objects: Query<(&TerrainCollider, &mut Velocity, &mut Transform)>,
    voxels: Res<ChunkMap<Voxel, ChunkShape>>,
) {
    println!();
    for (c, mut v, mut t) in &mut objects {
        let aabb_half_extents = c.aabb_half_extents();
        let mut displacement = Vec3::ZERO;
        let mut time_remaining = SIMULATION_STEP;
        for _ in 0..3 {
            // at most 3 iterations, one for each axis
            let center = t.translation;
            let min = Vec3::min(
                center - aabb_half_extents,
                center - aabb_half_extents + **v * time_remaining,
            )
            .floor()
            .as_ivec3();
            let max = Vec3::max(
                center + aabb_half_extents,
                center + aabb_half_extents + **v * time_remaining,
            )
            .ceil()
            .as_ivec3()
                - 1;
            if let Some(collision) = iproduct!(min.x..=max.x, min.y..=max.y, min.z..=max.z)
                .map(|(x, y, z)| IVec3::new(x, y, z))
                .filter(|voxel| voxels.voxel_at(*voxel).is_some_and(Voxel::collidable))
                // .inspect(|voxel| println!("interested in {}", voxel))
                .flat_map(|voxel| {
                    c.cube_collision(
                        &center,
                        &v,
                        &Cube {
                            center: voxel.as_vec3() + Vec3::splat(0.5),
                            radius: 0.5,
                        },
                        time_remaining,
                    )
                })
                .inspect(|collision| println!("{:?}", collision))
                .min_by(|a, b| a.time.total_cmp(&b.time))
            {
                displacement += **v * collision.time;
                let p_v = **v;
                // cancel velocity in the normal direction
                **v -= Vec3::dot(p_v, collision.normal) * collision.normal;
                time_remaining -= collision.time;
            } else {
                displacement += time_remaining * **v;
                break; // no more steps needed
            }
        }
        println!("{} += {}", t.translation, displacement);
        t.translation += displacement;
    }
}

fn apply_drag(mut objects: Query<(&Drag, &mut Acceleration)>) {
    for (d, mut a) in &mut objects {
        **a *= **d * SIMULATION_STEP;
    }
}

#[derive(Clone, Copy, Component, Default, Debug, Deref, DerefMut)]
pub struct Acceleration(pub Vec3);

#[derive(Clone, Copy, Component, Default, Debug, Deref, DerefMut)]
pub struct Velocity(pub Vec3);

#[derive(Component, Debug)]
/// A physics object which can collide with terrain.
pub enum TerrainCollider {
    Cylinder { radius: f32, half_height: f32 },
}

impl TerrainCollider {
    /// returns the maximum block offset that could collide
    fn aabb_half_extents(&self) -> Vec3 {
        match self {
            TerrainCollider::Cylinder {
                radius,
                half_height,
            } => Vec3::new(*radius, *half_height, *radius),
        }
    }

    fn cube_collision(
        &self,
        center: &Vec3,
        velocity: &Vec3,
        cube: &Cube,
        simulation_step: f32,
    ) -> Option<Collision> {
        match self {
            TerrainCollider::Cylinder {
                radius,
                half_height,
            } => cylinder_cube_collision(
                &Cylinder {
                    center: *center,
                    radius: *radius,
                    half_height: *half_height,
                },
                velocity,
                &cube,
                simulation_step,
            ),
        }
    }
}

#[derive(Clone, Copy, Component, Debug, Deref, DerefMut)]
pub struct Drag(pub f32);

// Represents a capped cylinder aligned to the y-axis.
#[derive(Clone, Debug)]
struct Cylinder {
    center: Vec3,
    radius: f32,
    half_height: f32,
}

/// Represents an axis-aligned cube (voxel).
#[derive(Debug)]
struct Cube {
    center: Vec3,
    radius: f32,
}

/// Information about a collision.
#[derive(PartialEq)]
struct Collision {
    time: f32,
    normal: Vec3,
}

impl std::fmt::Debug for Collision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // truncate the values to make snapshots cross-platform (since floating point errors are terrible)
        let precision = f.precision().unwrap_or(5);
        write!(
            f,
            "Collision {{\n\ttime: {:.*},\n\tnormal: Vec3(\n\t\tx: {:.*},\n\t\ty: {:.*},\n\t\tz: {:.*} \n\t),\n}}",
            precision,
            self.time,
            precision,
            self.normal.x,
            precision,
            self.normal.y,
            precision,
            self.normal.z
        )
    }
}

/// Returns whether the given cylinder and cube intersect.
fn cylinder_cube_intersection(cylinder: &Cylinder, cube: &Cube) -> bool {
    if (cylinder.center.y - cube.center.y).abs() > cylinder.half_height + cube.radius {
        // intersection is not possible, cylinder is above/below cube
        false
    } else {
        // project the cylinder and cube onto the xz axis and check for square/circle intersection
        // see: https://stackoverflow.com/questions/401847/circle-rectangle-collision-detection-intersection
        let cx = (cylinder.center.x - cube.center.x).abs();
        let cz = (cylinder.center.z - cube.center.z).abs();
        if cx.max(cz) > cylinder.radius + cube.radius {
            // too far on either x or z axis
            false
        } else if cx.min(cz) <= cube.radius {
            // close enough to guarantee intersection
            true
        } else {
            // check whether the circle clips the corner
            let corner_dist_sq = (cx - cube.radius).powi(2) + (cz - cube.radius).powi(2);
            corner_dist_sq <= cylinder.radius.powi(2)
        }
    }
}

/// Detects collision between a moving cylinder and stationary cube. Note that it is possible to clip if the cylinder
/// is moving too fast and/or the simulation step is too large!
fn cylinder_cube_collision(
    cylinder: &Cylinder,
    velocity: &Vec3,
    cube: &Cube,
    simulation_step: f32,
) -> Option<Collision> {
    // number of iterations for iterative collision resolver
    const MAX_ITER: usize = 30;
    // convergence threshold exponent (`1e-(THRESHOLD_POW)` is the threshold)
    const THRESHOLD_POW: i32 = 6;

    let threshold: f32 = 10.0_f32.powi(-THRESHOLD_POW);

    // finds (approximately) the last time in [0, simulation_time] when the cylinder and cube do not intersect,
    // assuming the cylinder moves with constant [velocity] and the two are initially not intersecting.

    if cylinder_cube_intersection(&cylinder, &cube) {
        // panic!("cylinder and cube are initially intersecting -- this is a physics bug!");
        // no way to know the normal!
        // todo: should we find the nearest face and set the normal to resolve the collision that way?
        return None;
    }

    if !cylinder_cube_intersection(
        &Cylinder {
            center: cylinder.center + *velocity * simulation_step,
            ..*cylinder
        },
        &cube,
    ) {
        // the final position of the cylinder does not collide with the cube
        None
    } else {
        // binary search for the last time of non-intersection
        // invariant: low is non-intersecting, high is intersecting.
        let (mut low, mut high) = (0., simulation_step);
        for _ in 0..MAX_ITER {
            if high - low < threshold {
                // we've converged!
                break;
            }
            let mid = (low + high) / 2.;
            let translated_cyl = Cylinder {
                center: cylinder.center + *velocity * mid,
                ..*cylinder
            };
            if cylinder_cube_intersection(&translated_cyl, &cube) {
                high = mid;
            } else {
                low = mid;
            }
        }

        let center = cylinder.center + *velocity * low;

        // To determine the normal, we just nudge [center] a bit in each direction of its velocity, and check whether
        // it collides post-nudge.

        let normal = velocity
            .signum()
            .to_array()
            .into_iter()
            .enumerate()
            .find_map(|(axis, sgn)| {
                let nudge = Vec3::AXES[axis] * sgn * high;
                let nudged_cyl = Cylinder {
                    center: center + nudge,
                    ..*cylinder
                };
                cylinder_cube_intersection(&nudged_cyl, &cube)
                    .then(|| Vec3::AXES[axis] * -sgn)
            })
            .expect("cylinder and cube collide, a normal could not be determined! this is a physics bug.");

        Some(Collision { time: low, normal })
    }
}

#[cfg(test)]
mod test_cylinder_cube_collisions {
    use bevy::prelude::Vec3;
    use insta::assert_debug_snapshot;

    use super::*;
    use crate::voxel::physics::cylinder_cube_collision;

    const CYLINDER: Cylinder = Cylinder {
        center: Vec3::ZERO,
        radius: 0.4,
        half_height: 0.9,
    };

    #[test]
    fn test_move_pos_y() {
        let cube = Cube {
            center: Vec3::Y * 1.5,
            radius: 0.5,
        };

        assert!(cylinder_cube_collision(&CYLINDER, &Vec3::Y, &cube, 0.1 - 1e-6).is_none());
        assert_debug_snapshot!(
            cylinder_cube_collision(&CYLINDER, &Vec3::Y, &cube, 1.0),
            @r###"
        Some(
            Collision {
            	time: 0.10000,
            	normal: Vec3(
            		x: -0.00000,
            		y: -1.00000,
            		z: -0.00000 
            	),
            },
        )
        "###
        );
    }

    #[test]
    fn test_move_neg_y() {
        let cube = Cube {
            center: Vec3::NEG_Y * 1.5,
            radius: 0.5,
        };

        assert!(cylinder_cube_collision(&CYLINDER, &Vec3::NEG_Y, &cube, 0.05).is_none());
        assert_debug_snapshot!(
            cylinder_cube_collision(&CYLINDER, &Vec3::NEG_Y, &cube, 1.0),
            @r###"
        Some(
            Collision {
            	time: 0.10000,
            	normal: Vec3(
            		x: 0.00000,
            		y: 1.00000,
            		z: 0.00000 
            	),
            },
        )
        "###
        );
    }

    #[test]
    fn test_move_pos_x() {
        let cube = Cube {
            center: Vec3::X * 1.5,
            radius: 0.5,
        };

        assert!(cylinder_cube_collision(&CYLINDER, &Vec3::X, &cube, 0.6 - 1e-6).is_none());
        assert_debug_snapshot!(
            cylinder_cube_collision(&CYLINDER, &Vec3::X, &cube, 1.0),
            @r###"
        Some(
            Collision {
            	time: 0.60000,
            	normal: Vec3(
            		x: -1.00000,
            		y: -0.00000,
            		z: -0.00000 
            	),
            },
        )
        "###
        );
    }

    #[test]
    fn test_move_neg_x() {
        let cube = Cube {
            center: Vec3::NEG_X * 1.5,
            radius: 0.5,
        };

        assert!(cylinder_cube_collision(&CYLINDER, &Vec3::NEG_X, &cube, 0.6 - 1e-6).is_none());
        assert_debug_snapshot!(
            cylinder_cube_collision(&CYLINDER, &Vec3::NEG_X, &cube, 1.0),
            @r###"
        Some(
            Collision {
            	time: 0.60000,
            	normal: Vec3(
            		x: 1.00000,
            		y: 0.00000,
            		z: 0.00000 
            	),
            },
        )
        "###
        );
    }

    #[test]
    fn test_move_pos_z() {
        let cube = Cube {
            center: Vec3::Z * 1.5,
            radius: 0.5,
        };

        assert!(cylinder_cube_collision(&CYLINDER, &Vec3::Z, &cube, 0.6 - 1e-6).is_none());
        assert_debug_snapshot!(
            cylinder_cube_collision(&CYLINDER, &Vec3::Z, &cube, 1.0),
            @r###"
        Some(
            Collision {
            	time: 0.60000,
            	normal: Vec3(
            		x: -0.00000,
            		y: -0.00000,
            		z: -1.00000 
            	),
            },
        )
        "###
        );
    }

    #[test]
    fn test_move_neg_z() {
        let cube = Cube {
            center: Vec3::NEG_Z * 1.5,
            radius: 0.5,
        };

        assert!(cylinder_cube_collision(&CYLINDER, &Vec3::NEG_Z, &cube, 0.6 - 1e-6).is_none());
        assert_debug_snapshot!(
            cylinder_cube_collision(&CYLINDER, &Vec3::NEG_Z, &cube, 1.0),
            @r###"
        Some(
            Collision {
            	time: 0.60000,
            	normal: Vec3(
            		x: 0.00000,
            		y: 0.00000,
            		z: 1.00000 
            	),
            },
        )
        "###
        );
    }
}
