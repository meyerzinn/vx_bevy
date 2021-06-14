use std::collections::VecDeque;

use bevy::{math::IVec2, prelude::*, render::pipeline::PrimitiveTopology, utils::HashMap};
use building_blocks::prelude::*;
use heron::prelude::*;

use crate::{config::GlobalConfig, voxel::Voxel, Player};

use super::{
    chunk2global, chunk_extent, global2chunk,
    worldgen::{NoiseTerrainGenerator, TerrainGenerator},
    ChunkMeshInfo,
};

pub type ChunkEntityMap = HashMap<IVec2, Entity>;

/// A component tracking the current loading state of a chunk.
pub enum ChunkLoadState {
    Load,   // Chunk needs to be loaded from disk.
    Unload, // Chunk needs to be saved to disk and unloaded.
    //Despawn, // Chunk will be despawned on next frame.
    Generate, // Chunk wasn't generated beforehand and needs to be generated by the worldgen.
    Done,     // Chunk is done loading.
}

pub(crate) struct ChunkSpawnRequest(IVec2);
pub(crate) struct ChunkDespawnRequest(IVec2, Entity);

pub(crate) struct ChunkLoadRequest(Entity);

/// An event signaling that a chunk and its data have finished loading and are ready to be displayed.
pub struct ChunkReadyEvent(pub IVec2, pub Entity);

/// A component describing a chunk.
pub struct Chunk {
    pub pos: IVec2,
    pub block_data: Array3x1<Voxel>,
}

#[derive(Bundle)]
pub struct ChunkDataBundle {
    pub transform: Transform,
    pub global_transform: GlobalTransform,
    pub chunk: Chunk,
    pub rigid_body: RigidBody,
    pub collision_shape: CollisionShape,
    pub mesh_info: ChunkMeshInfo,
}

/// Handles the visibility checking of the currently loaded chunks around the player.
/// This will accordingly emit [`ChunkSpawnRequest`] events for chunks that need to be loaded since they entered the player's view distance and [`ChunkDespawnRequest`] for
/// chunks out of the player's view distance.
pub(crate) fn update_visible_chunks(
    player: Query<(&Transform, &Player)>,
    world: Res<ChunkEntityMap>,
    config: Res<GlobalConfig>,
    mut load_radius_chunks: bevy::ecs::system::Local<Vec<IVec2>>,
    mut spawn_requests: EventWriter<ChunkSpawnRequest>,
    mut despawn_requests: EventWriter<ChunkDespawnRequest>,
) {
    if let Ok((transform, _)) = player.single() {
        let pos = global2chunk(transform.translation);

        for dx in -config.render_distance..=config.render_distance {
            for dy in -config.render_distance..=config.render_distance {
                if dx.pow(2) + dy.pow(2) >= config.render_distance.pow(2) {
                    continue;
                };

                let chunk_pos = pos + (dx, dy).into();
                if !world.contains_key(&chunk_pos) {
                    load_radius_chunks.push(chunk_pos);
                }
            }
        }

        load_radius_chunks.sort_by_key(|a| (a.x.pow(2) + a.y.pow(2)));

        spawn_requests.send_batch(
            load_radius_chunks
                .drain(..)
                .map(|c| ChunkSpawnRequest(c.clone())),
        );

        for key in world.keys() {
            let delta = *key - pos;
            let entity = world.get(key).unwrap().clone();
            if delta.x.abs().pow(2) + delta.y.abs().pow(2) > config.render_distance.pow(2) {
                despawn_requests.send(ChunkDespawnRequest(key.clone(), entity));
            }
        }
    }
}

pub(crate) fn create_chunks(
    mut commands: Commands,
    mut spawn_events: EventReader<ChunkSpawnRequest>,
    mut world: ResMut<ChunkEntityMap>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    for creation_request in spawn_events.iter() {
        let entity = commands
            .spawn_bundle(ChunkDataBundle {
                transform: Transform::from_translation(chunk2global(creation_request.0)),
                chunk: Chunk {
                    pos: creation_request.0,
                    block_data: Array3x1::fill(chunk_extent().padded(1), Voxel::default()),
                },
                global_transform: Default::default(),
                rigid_body: RigidBody::Static,
                collision_shape: CollisionShape::Sphere { radius: 16.0 },
                mesh_info: ChunkMeshInfo {
                    fluid_mesh: meshes.add(Mesh::new(PrimitiveTopology::TriangleList)),
                    chunk_mesh: meshes.add(Mesh::new(PrimitiveTopology::TriangleList)),
                },
            })
            .insert(ChunkLoadState::Load)
            .id();

        world.insert(creation_request.0, entity);
    }
}

//todo: parallelize this.
//todo: run this on the IOTaskPool
/// Loads from disk the chunk data of chunks with a current load state of [`ChunkLoadState::Load`].
/// If the chunk wasn't generated, the [`ChunkLoadState`] of the chunk is set to [`ChunkLoadState::Generate`].
pub(crate) fn load_chunk_data(
    mut chunks: Query<(&mut ChunkLoadState, Entity), Added<Chunk>>,
    mut gen_requests: ResMut<VecDeque<ChunkLoadRequest>>,
) {
    for (mut load_state, entity) in chunks.iter_mut() {
        match *load_state {
            ChunkLoadState::Load => {
                *load_state = ChunkLoadState::Generate;
                gen_requests.push_front(ChunkLoadRequest(entity));
            }
            _ => continue,
        }
    }
}

/// Marks the load state of all chunk that are queued to be unloaded as [`ChunkLoadState::Unload`]
pub(crate) fn prepare_for_unload(
    mut despawn_events: EventReader<ChunkDespawnRequest>,
    mut chunks: Query<&mut ChunkLoadState>,
) {
    for despawn_event in despawn_events.iter() {
        if let Ok(mut load_state) = chunks.get_mut(despawn_event.1) {
            *load_state = ChunkLoadState::Unload;
        }
    }
}

/// Destroys all the chunks that have a load state of [`ChunkLoadState::Unload`]
pub(crate) fn destroy_chunks(
    mut commands: Commands,
    mut world: ResMut<ChunkEntityMap>,
    chunks: Query<(&Chunk, &ChunkLoadState)>,
) {
    for (chunk, load_state) in chunks.iter() {
        match load_state {
            ChunkLoadState::Unload => {
                let entity = world.remove(&chunk.pos).unwrap();
                commands.entity(entity).despawn_recursive();
            }
            _ => {}
        }
    }
}

pub(crate) fn generate_chunks(
    mut query: Query<(&mut Chunk, &mut ChunkLoadState)>,
    mut gen_requests: ResMut<VecDeque<ChunkLoadRequest>>,
    config: Res<GlobalConfig>,
    gen: Res<NoiseTerrainGenerator>,
) {
    for _ in 0..(config.render_distance / 2) {
        if let Some(ev) = gen_requests.pop_back() {
            if let Ok((mut data, mut load_state)) = query.get_mut(ev.0) {
                gen.generate(data.pos, &mut data.block_data);
                *load_state = ChunkLoadState::Done;
            }
        }
    }
}

pub(crate) fn mark_chunks_ready(
    mut ready_events: EventWriter<ChunkReadyEvent>,
    chunks: Query<(&Chunk, &ChunkLoadState, Entity), Changed<ChunkLoadState>>,
) {
    for (chunk, load_state, entity) in chunks.iter() {
        match load_state {
            ChunkLoadState::Done => ready_events.send(ChunkReadyEvent(chunk.pos, entity)),
            _ => {}
        }
    }
}
