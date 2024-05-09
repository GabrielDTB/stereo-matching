use anyhow::Context;
use anyhow::{ensure, Result};
use clap::Parser;
use image::io::Reader as ImageReader;
use image::ImageBuffer;
use image::Luma;
use rayon::prelude::*;

fn point_to_index(x: i64, y: i64, width: i64) -> usize {
    (x + (y * width)) as usize
}

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

fn calculate_disparity(
    left: &[u8],
    right: &[u8],
    padding: i64,
    height: i64,
    left_width: i64,
    right_width: i64,
    y: i64,
    left_x: i64,
) -> u64 {
    (0..right_width as i64)
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

fn calculate_disparities_across_y(
    left: &[u8],
    right: &[u8],
    padding: i64,
    height: i64,
    left_width: i64,
    right_width: i64,
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
                y,
                left_x,
            )
        })
        .collect::<Vec<_>>()
}

fn calculate_disparities(
    left: &ImageBuffer<Luma<u8>, Vec<u8>>,
    right: &ImageBuffer<Luma<u8>, Vec<u8>>,
    window_size: i64,
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
                y,
            )
        })
        .collect::<Vec<_>>())
}

fn scale_disparities(disparities: &Vec<u64>) -> Vec<u8> {
    disparities
        .iter()
        .map(|&i| std::cmp::min(i * 4, 255) as u8)
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

fn get_ground_truth_disparities(path: String) -> Result<Vec<u8>> {
    Ok(ImageReader::open(path)?.decode()?.into_luma8().into_vec())
}

fn save_disparity_map_as_image(
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

fn calculate_error_rate(disparities: &Vec<u64>, ground_truth: &Vec<u8>) -> f64 {
    let error_count: u64 = disparities
        .iter()
        .zip(
            ground_truth
                .iter()
                .map(|&x| (x as f64 / 4.0).round() as u64),
        )
        .map(|(&a, b)| if a.abs_diff(b) > 1 { 1 } else { 0 })
        .sum();
    error_count as f64 / disparities.len() as f64
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

    #[arg(short = 'g', long, value_name = "PATH")]
    ground_truth: Option<String>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let output_path = args.output;
    let ground_truth = match args.ground_truth {
        Some(path) => Some(get_ground_truth_disparities(path)?),
        _ => None,
    };
    let window_size = args.window_size;
    let (left_image, right_image) =
        open_images(args.image_left.as_str(), args.image_right.as_str())?;

    let disparities = calculate_disparities(&left_image, &right_image, args.window_size)?;

    if let Some(ground_truth) = ground_truth {
        let error_rate = calculate_error_rate(&disparities, &ground_truth);
        println!(
            "Error rate with window size {window_size}:\t{:.2}%",
            error_rate * 100.0
        );
    }

    let scaled_disparities = scale_disparities(&disparities);
    save_disparity_map_as_image(
        scaled_disparities,
        left_image.width(),
        left_image.height(),
        &output_path,
    )?;

    Ok(())
}
