use anyhow::{bail, Context};
use anyhow::{ensure, Result};
use clap::Parser;
use image::io::Reader as ImageReader;
use image::ImageBuffer;
use image::Luma;
use rayon::prelude::*;

const MAX_DISPARITY: i64 = 64;

fn point_to_index(x: i64, y: i64, width: i64) -> usize {
    (x + (y * width)) as usize
}

/// Calculates the absolute difference between the pixels at two coordinates.
/// If a point is out of range of the image, 0 is used for that points's value.
fn calculate_ad(
    left: &[u8],
    right: &[u8],
    height: i64,
    left_width: i64,
    right_width: i64,
    y: i64,
    left_x: i64,
    right_x: i64,
) -> u64 {
    match (
        y >= 0 && y < height,
        left_x >= 0 && left_x < left_width,
        right_x >= 0 && right_x < right_width,
    ) {
        (true, true, false) => left[point_to_index(left_x, y, left_width)] as u64,
        (true, false, true) => right[point_to_index(right_x, y, right_width)] as u64,
        (true, true, true) => left[point_to_index(left_x, y, left_width)]
            .abs_diff(right[point_to_index(right_x, y, right_width)])
            as u64,
        _ => 0,
    }
}

/// Calculates the SAD between two windows.
fn calculate_sad(
    left: &[u8],
    right: &[u8],
    padding: i64,
    height: i64,
    left_width: i64,
    right_width: i64,
    y: i64,
    left_x: i64,
    right_x: i64,
) -> u64 {
    let mut sad = 0;
    for (final_left_x, final_right_x) in
        ((left_x - padding)..(left_x + padding)).zip((right_x - padding)..(right_x + padding))
    {
        for final_y in (y - padding)..(y + padding) {
            sad += calculate_ad(
                left,
                right,
                height,
                left_width,
                right_width,
                final_y,
                final_left_x,
                final_right_x,
            );
        }
    }
    sad
}

/// Calculates the disparity for a pixel on the left image over a strip on the right image.
fn calculate_disparity(
    left: &[u8],
    right: &[u8],
    padding: i64,
    height: i64,
    left_width: i64,
    right_width: i64,
    max_disparity: i64,
    y: i64,
    left_x: i64,
) -> u64 {
    (std::cmp::max(0, left_x - max_disparity)..left_x as i64)
        .fold((u64::MAX, 0), |(best_sad, best_disparity), right_x| {
            let sad = calculate_sad(
                left,
                right,
                padding,
                height,
                left_width,
                right_width,
                y,
                left_x,
                right_x,
            );
            if sad < best_sad {
                (sad, left_x.abs_diff(right_x))
            } else {
                (best_sad, best_disparity)
            }
        })
        .1
}

/// Calculates disparities for a strip of pixels on the left image.
fn calculate_disparities_across_y(
    left: &[u8],
    right: &[u8],
    padding: i64,
    height: i64,
    left_width: i64,
    right_width: i64,
    max_disparity: i64,
    y: i64,
) -> Vec<u64> {
    (0..left_width as i64)
        .map(|left_x| {
            calculate_disparity(
                left,
                right,
                padding,
                height,
                left_width,
                right_width,
                max_disparity,
                y,
                left_x,
            )
        })
        .collect::<Vec<_>>()
}

/// Calculates disparities for the entire left image.
fn calculate_disparities(
    left: &ImageBuffer<Luma<u8>, Vec<u8>>,
    right: &ImageBuffer<Luma<u8>, Vec<u8>>,
    window_size: i64,
    max_disparity: i64,
) -> Result<Vec<u64>> {
    ensure!(
        left.height() == right.height(),
        "Image heights must be equal."
    );
    ensure!(window_size % 2 == 1, "Window size must be odd.");

    let padding = window_size / 2;
    let height = left.height() as i64;
    let [left_width, right_width] = [left, right].map(|i| i.width() as i64);

    let [left_buffer, right_buffer] = [left, right].map(|i| i.as_raw());

    Ok((0..height as i64)
        .into_par_iter()
        .flat_map(|y| {
            calculate_disparities_across_y(
                &left_buffer,
                &right_buffer,
                padding,
                height,
                left_width,
                right_width,
                max_disparity,
                y,
            )
        })
        .collect::<Vec<_>>())
}

fn scale_disparities_to_pixels(disparities: &Vec<u64>, max_disparity: i64) -> Vec<u8> {
    disparities
        .iter()
        .map(|&i| i as f64)
        .map(|f| (f * (255.0 / max_disparity as f64)).round() as u8)
        .collect::<Vec<_>>()
}

fn open_images(
    left_path: &str,
    right_path: &str,
) -> Result<(
    ImageBuffer<Luma<u8>, Vec<u8>>,
    ImageBuffer<Luma<u8>, Vec<u8>>,
)> {
    Ok((
        ImageReader::open(left_path)?.decode()?.into_luma8(),
        ImageReader::open(right_path)?.decode()?.into_luma8(),
    ))
}

fn get_ground_truth_disparities(path: String, max_disparity: i64) -> Result<Vec<u64>> {
    Ok(ImageReader::open(path)?
        .decode()?
        .into_luma8()
        .into_vec()
        .into_iter()
        .map(|v| (v as f64 * (max_disparity as f64 / 255.0)).round() as u64)
        .collect::<Vec<_>>())
}

fn save_vec_as_image(
    scaled_disparities: Vec<u8>,
    width: u32,
    height: u32,
    filename: &str,
) -> Result<()> {
    let image: ImageBuffer<Luma<u8>, Vec<u8>> =
        ImageBuffer::from_vec(width, height, scaled_disparities)
            .context("Failed to make image buffer from disparity map.")?;
    Ok(image.save_with_format(filename, image::ImageFormat::Png)?)
}

/// Calculates the error map of computed disparities to the ground truth.
/// The disparity values should be real disparities, not pixel values.
/// A difference up to plus or minus 1 is deemed okay.
fn calculate_error_map(disparities: &Vec<u64>, ground_truth: &Vec<u64>) -> Vec<bool> {
    disparities
        .iter()
        .zip(ground_truth)
        .map(|(&a, &b)| if a.abs_diff(b) > 1 { true } else { false })
        .collect::<Vec<_>>()
}

fn calculate_error_rate(
    disparities: &Vec<u64>,
    ground_truth: &Vec<u64>,
    valid_pixels: &Vec<bool>,
) -> f64 {
    calculate_error_map(disparities, ground_truth)
        .iter()
        .zip(valid_pixels)
        .map(|(&b, &t)| if b && t { 1 } else { 0 })
        .sum::<u64>() as f64
        / (disparities.len() - valid_pixels.iter().filter(|&a| !a).count()) as f64
}

/// Calculates the map for points that can possibly have the correct disparity value,
/// accounting for the left margin.
fn calculate_in_bounds_map(ground_truth: &Vec<u64>, width: u32) -> Vec<bool> {
    ground_truth
        .chunks_exact(width as usize)
        .map(|i| i.into_iter().enumerate())
        .flatten()
        .map(|(x, &disp)| if (x as u64) < disp { false } else { true })
        .collect::<Vec<_>>()
}

/// TODO 2: Make function to calculate the not_occluded map.
/// The algorithm for this map is to traverse the computed disparity map from right to left,
/// mapping each point (with disparity) to a corresponding array.
/// If the array entry that a point gets mapped to already has been mapped to, then the
/// current point must be occluded.
/// For efficiency, we may use the calculate_in_bounds_map to reduce the number of points we traverse.
/// Or, we might combine the two methods.

fn calculate_valid_pixels(ground_truth: &Vec<u64>, width: u32) -> Vec<bool> {
    let mut target_pixels = vec![false; ground_truth.len()];
    let mut valid_pixels = vec![true; ground_truth.len()];
    let in_bounds = calculate_in_bounds_map(ground_truth, width);

    for i in (0..valid_pixels.len()).rev() {
        if !in_bounds[i] {
            valid_pixels[i] = false;
        } else {
            let location = i - ground_truth[i] as usize;
            if target_pixels[location] {
                valid_pixels[i] = false;
            } else {
                target_pixels[location] = true;
            }
        }
    }

    valid_pixels
}

fn calculate_valid_pixels_using_iterators(ground_truth: &Vec<u64>, width: u32) -> Vec<bool> {
    let width = width as usize;
    ground_truth
        .chunks_exact(width)
        .map(|disparities| {
            disparities
                .into_iter()
                .map(|&disparity| disparity as usize)
                .enumerate()
                .rev()
        })
        .map(|positions_disparities| {
            let mut targets = vec![false; width];
            positions_disparities
                .map(|(position, disparity)| {
                    if position >= disparity && !targets[position - disparity] {
                        targets[position - disparity] = true;
                        true
                    } else {
                        false
                    }
                })
                .collect::<Vec<_>>()
        })
        .map(|validities| validities.into_iter().rev())
        .flatten()
        .collect::<Vec<_>>()
}

fn census_transform(image: &[u8], padding: i64, height: i64, width: i64, x: i64, y: i64) -> u64 {
    let mut descriptor: u64 = 0;

    for final_x in (x - padding)..(x + padding) {
        for final_y in (y - padding)..(y + padding) {
            if x == final_x && y == final_y {
                continue;
            } else {
                let neighbor_value = calculate_pixel_value(image, height, width, final_x, final_y);
                descriptor <<= 1;
                match neighbor_value >= image[point_to_index(x, y, width)] {
                    true => descriptor |= 1,
                    false => (),
                }
            }
        }
    }
    descriptor
}

fn rank_transform(image: &[u8], padding: i64, height: i64, width: i64, x: i64, y: i64) -> u64 {
    let mut neighborhood = vec![];

    for final_x in (x - padding)..(x + padding) {
        for final_y in (y - padding)..(y + padding) {
            let pixel_value = calculate_pixel_value(image, height, width, final_x, final_y);
            neighborhood.push(pixel_value);
        }
    }

    neighborhood.sort();

    let central_pixel_value = calculate_pixel_value(image, height, width, x, y);
    let rank = neighborhood
        .iter()
        .position(|&val| val == central_pixel_value)
        .unwrap_or(0);

    rank as u64
}

fn calculate_pixel_value(image: &[u8], height: i64, width: i64, x: i64, y: i64) -> u8 {
    if y >= 0 && y < height && x >= 0 && x < width {
        image[point_to_index(x, y, width)]
    } else {
        0
    }
}

/// Calculates disparity map between two images
#[derive(Parser, Debug)]
#[command(about, long_about = None)]
struct Args {
    /// Path to left image
    #[arg(index = 1, value_name = "PATH")]
    image_left: String,

    /// Path to right image
    #[arg(index = 2, value_name = "PATH")]
    image_right: String,

    /// Window size for SAD calculations
    #[arg(short = 'w', long, value_name = "SIZE", default_value_t = 5)]
    window_size: i64,

    /// Path to save output image
    #[arg(short = 'o', long, value_name = "PATH", default_value_t = String::from("disparities.png"))]
    output: String,

    /// Path to ground truth disparity map
    #[arg(short = 'g', long, value_name = "PATH")]
    ground_truth: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let output_path = args.output;
    let (output_name, output_extension) = output_path
        .rsplit_once('.')
        .context("Output path must have file extension, none was found.")?;
    let ground_truth = match args.ground_truth {
        Some(path) => Some(get_ground_truth_disparities(path, MAX_DISPARITY)?),
        _ => {
            println!("No ground truth given: some outputs will not be present.");
            None
        }
    };
    let window_size = args.window_size;
    let (left_image, right_image) =
        open_images(args.image_left.as_str(), args.image_right.as_str())?;

    let disparities =
        calculate_disparities(&left_image, &right_image, args.window_size, MAX_DISPARITY)?;
    save_vec_as_image(
        scale_disparities_to_pixels(&disparities, MAX_DISPARITY),
        left_image.width(),
        left_image.height(),
        &output_path,
    )?;

    if let Some(ground_truth) = ground_truth {
        let valid_pixels = calculate_valid_pixels(&ground_truth, left_image.width());
        let error_rate = calculate_error_rate(&disparities, &ground_truth, &valid_pixels);
        println!(
            "Error rate with window size {window_size}:\t{:.2}%",
            error_rate * 100.0
        );
        let error_map = calculate_error_map(&disparities, &ground_truth);

        save_vec_as_image(
            error_map
                .iter()
                .map(|&b| if b { 255 } else { 0 })
                .collect::<Vec<_>>(),
            left_image.width(),
            left_image.height(),
            &format!("{output_name}.error_map.{output_extension}"),
        )?;

        let bounds = calculate_in_bounds_map(&ground_truth, left_image.width());
        save_vec_as_image(
            bounds
                .iter()
                .map(|&b| if b { 255 } else { 0 })
                .collect::<Vec<_>>(),
            left_image.width(),
            left_image.height(),
            &format!("{output_name}.bounds_map.{output_extension}"),
        )?;

        save_vec_as_image(
            valid_pixels
                .iter()
                .map(|&b| if b { 255 } else { 0 })
                .collect::<Vec<_>>(),
            left_image.width(),
            left_image.height(),
            &format!("{output_name}.validity_map.{output_extension}"),
        )?;
    }

    Ok(())
}
