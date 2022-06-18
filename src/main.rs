#![allow(dead_code)]

use fastrand::Rng;
use fxhash::FxHashSet as HashSet;
use log::error;
use pixels::{Error, Pixels, SurfaceTexture};
use winit::dpi::LogicalSize;
use winit::event::{Event, VirtualKeyCode};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::WindowBuilder;
use winit_input_helper::WinitInputHelper;

const WIDTH: usize = 512;
const HEIGHT: usize = 512;
const W_SLICE: usize = 14;
const H_SLICE: usize = 26;
const NUM_REGIONS: usize = W_SLICE * H_SLICE;
const DEFAULT_SCALE: f64 = 1.0;

fn main() -> Result<(), Error> {
    env_logger::init();
    let event_loop = EventLoop::new();
    let mut input = WinitInputHelper::new();
    let window = {
        let size = LogicalSize::new(WIDTH as f64 * DEFAULT_SCALE, HEIGHT as f64 * DEFAULT_SCALE);
        WindowBuilder::new()
            .with_title("Gerrymander")
            .with_inner_size(size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let mut pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(WIDTH as u32, HEIGHT as u32, surface_texture)?
    };
    let mut world = World::new();

    event_loop.run(move |event, _, control_flow| {
        // Draw the current frame
        if let Event::RedrawRequested(_) = event {
            world.draw(pixels.get_frame());
            if pixels
                .render()
                .map_err(|e| error!("pixels.render() failed: {}", e))
                .is_err()
            {
                *control_flow = ControlFlow::Exit;
                return;
            }
        }

        // Handle input events
        if input.update(&event) {
            // Close events
            if input.key_pressed(VirtualKeyCode::Escape) || input.quit() {
                *control_flow = ControlFlow::Exit;
                return;
            }

            // Resize the window
            if let Some(size) = input.window_resized() {
                pixels.resize_surface(size.width, size.height);
            }

            // Update internal state and request a redraw
            world.update();
            window.request_redraw();
        }
    });
}

type Rgb = (u8, u8, u8);

fn gen_colors(mut n: usize) -> Vec<Rgb> {
    let rng = Rng::with_seed(n as u64);
    let mut colors = HashSet::default();
    while n > 0 {
        let color = (rng.u8(..), rng.u8(..), rng.u8(..));
        if colors.insert(color) {
            n -= 1;
        }
    }
    colors.into_iter().collect()
}

/// Representation of the application state. In this example, a box will bounce around the screen.
struct World {
    cells: Vec<usize>,
    colors: Vec<Rgb>,
    rng: Rng,
    rand_threshold: f32,
    iterations: usize,
}

impl World {
    /// Create a new `World` instance.
    fn new() -> Self {
        let mut cells = vec![0; WIDTH * HEIGHT];
        let colors = gen_colors(NUM_REGIONS);
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                let ry = (y as f64 / HEIGHT as f64 * H_SLICE as f64).floor() as usize;
                let rx = (x as f64 / WIDTH as f64 * W_SLICE as f64).floor() as usize;
                let region_index = rx + ry * W_SLICE;
                cells[x + y * WIDTH] = region_index;
            }
        }
        Self {
            cells,
            colors,
            rng: Rng::with_seed(69),
            rand_threshold: 1.0,
            iterations: 0,
        }
    }

    fn region_cost(&self, region: usize) -> f32 {
        let pos: Vec<(f32, f32)> = self
            .cells
            .iter()
            .enumerate()
            .filter_map(|(i, &cell)| {
                (cell == region).then(|| ((i % WIDTH) as f32, (i / WIDTH) as f32))
            })
            .collect();
        let tuple_div = |(x, y), b| (x / b, y / b);
        let (cx, cy) = tuple_div(
            pos.iter().fold((0.0, 0.0), |(a, b), (c, d)| (a + c, b + d)),
            pos.len() as f32,
        );
        pos.into_iter()
            .map(|(x, y)| (x - cx) * (x - cx) + (y - cy) * (y - cy))
            .sum()
    }

    // fn region_cost(&self, region: usize) -> usize {
    //     self.cells
    //         .iter()
    //         .enumerate()
    //         .filter_map(|(i, &cell)| {
    //             (cell == region).then(|| {
    //                 let (x, y) = (i % WIDTH, i / WIDTH);
    //                 Neighbors::new(self, x, y).cost()
    //             })
    //         })
    //         .sum()
    // }

    // fn region_cost(&self, region: usize) -> usize {
    //     self.cells
    //         .iter()
    //         .enumerate()
    //         .filter(|(i, &cell)| {
    //             (cell == region) && {
    //                 let (x, y) = (i % WIDTH, i / WIDTH);
    //                 Neighbors::new(self, x, y).is_boundary_deterministic()
    //             }
    //         })
    //         .count()
    // }

    // fn region_cost(&self, region: usize) -> usize {
    //     self.cells
    //         .iter()
    //         .enumerate()
    //         .filter(|(i, &cell)| {
    //             let (x, y) = (i % WIDTH, i / WIDTH);
    //             cell != region
    //                 && Neighbors::new(self, x, y).has_region(region)
    //         })
    //         .count()
    // }

    fn update_pixel(&mut self) {
        loop {
            let test_x = self.rng.usize(0..WIDTH);
            let test_y = self.rng.usize(0..HEIGHT);
            assert!(self.get_index(test_x, test_y).is_some());
            let kernel = Neighbors::new(self, test_x, test_y);
            if let Some(color) = kernel.is_boundary(&self.rng) {
                if kernel.breaks_region() {
                    continue;
                }
                if self.rng.f32() < self.rand_threshold {
                    let cell = self.get_cell_mut(test_x, test_y).unwrap();
                    *cell = color;
                    return;
                }
                let old_color = self.get_cell(test_x, test_y).unwrap();
                let old_cost = self.region_cost(old_color);
                let cell = self.get_cell_mut(test_x, test_y).unwrap();
                *cell = color;
                let new_cost = self.region_cost(old_color);
                if new_cost < old_cost {
                    return;
                } else {
                    let cell = self.get_cell_mut(test_x, test_y).unwrap();
                    *cell = old_color;
                }
            }
        }
    }

    /// Update the `World` internal state.
    fn update(&mut self) {
        let iterations = 1000;
        for _ in 0..iterations {
            self.update_pixel();
        }
        self.iterations += 1;
        if self.iterations == WIDTH * HEIGHT / 100 {
            self.rand_threshold = 0.3;
            println!("Tighten!");
            self.colors[0] = (0, 0, 0);
        }
    }

    fn get_color(&self, i: usize) -> [u8; 4] {
        let (r, g, b) = self.colors[i];
        [r, g, b, 0xff]
    }

    fn get_index(&self, x: usize, y: usize) -> Option<usize> {
        (x < WIDTH && y < HEIGHT).then(|| x + y * WIDTH)
    }

    fn get_cell(&self, x: usize, y: usize) -> Option<usize> {
        self.get_index(x, y).map(|i| self.cells[i])
    }

    fn get_cell_mut(&mut self, x: usize, y: usize) -> Option<&mut usize> {
        self.get_index(x, y).map(|i| &mut self.cells[i])
    }

    /// Draw the `World` state to the frame buffer.
    ///
    /// Assumes the default texture format: `wgpu::TextureFormat::Rgba8UnormSrgb`
    fn draw(&self, frame: &mut [u8]) {
        for (pixel, &cell) in frame.chunks_exact_mut(4).zip(&self.cells) {
            pixel.copy_from_slice(&self.get_color(cell));
        }
    }
}

struct Neighbors(pub [Option<usize>; 9]);

impl Neighbors {
    fn new(w: &World, x: usize, y: usize) -> Self {
        let mut cells = [None; 9];
        let mut i = 0;
        for dy in -1..=1_isize {
            for dx in -1..=1_isize {
                let nx = checked_add(x, dx);
                let ny = checked_add(y, dy);
                if let (Some(nx), Some(ny)) = (nx, ny) {
                    cells[i] = w.get_cell(nx, ny);
                }
                i += 1;
            }
        }
        Self(cells)
    }
    fn is_same(&self, i: usize, j: usize) -> Option<bool> {
        self.0[i].and_then(|v1| self.0[j].map(|v2| v1 == v2))
    }
    /// Returns whether changing the middle cell could potentially divide a region in two
    fn breaks_region(&self) -> bool {
        let center = self.0[4];
        let is_region = self.0.map(|r| r == center);
        is_region[1] && is_region[7] || is_region[3] && is_region[5]
    }
    /// Returns whether the middle cell is on a boundary between regions
    fn is_boundary(&self, rng: &Rng) -> Option<usize> {
        let mut edges = [1, 3, 5, 7];
        rng.shuffle(&mut edges);
        for i in edges {
            if self.is_same(4, i) == Some(false) {
                return Some(self.0[i].unwrap());
            }
        }
        None
    }

    fn is_boundary_deterministic(&self) -> bool {
        for i in [1, 3, 5, 7] {
            if self.is_same(4, i) == Some(false) {
                return true;
            }
        }
        false
    }

    fn cost(&self) -> usize {
        (0..9)
            .filter(|&i| self.is_same(4, i) == Some(false))
            .count()
    }

    fn has_region(&self, r: usize) -> bool {
        self.0.iter().any(|&v| v == Some(r))
    }
}

const fn checked_add(lhs: usize, rhs: isize) -> Option<usize> {
    if rhs.is_negative() {
        lhs.checked_sub(rhs.wrapping_abs() as usize)
    } else {
        lhs.checked_add(rhs as usize)
    }
}

// const fn tuple_div(a: (f64, f64), b: f64) -> (f64, f64) {
//     (a.0 / b, a.1 / b)
// }
