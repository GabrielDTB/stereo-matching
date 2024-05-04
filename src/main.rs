use anyhow::Context;
use anyhow::{ensure, Result};
use image::io::Reader as ImageReader;
use image::ImageBuffer;
use image::Luma;
use rayon::prelude::*;

fn translate_coordinates(x: i64, y: i64, height: i64) -> usize {
    (x + (y * height)) as usize
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
    // for (offset_x, offset_y) in iproduct!(-padding..=padding, -padding..=padding) {
    // for offset_x in -padding..padding {
    for (final_left_x, final_right_x) in
        ((-padding + left_x)..(padding + left_x)).zip((-padding + right_x)..(padding + right_x))
    {
        for final_y in (-padding + y)..(padding + y) {
            let left_pixel;
            let right_pixel;

            if final_left_x < 0 || final_left_x >= left_width {
                left_pixel = 0;
            } else if final_y < 0 || final_y >= height {
                left_pixel = 0;
            } else {
                left_pixel = left[translate_coordinates(final_left_x, final_y, height)];
            }

            if final_right_x < 0 || final_right_x >= right_width {
                right_pixel = 0;
            } else if final_y < 0 || final_y >= height {
                right_pixel = 0;
            } else {
                right_pixel = right[translate_coordinates(final_right_x, final_y, height)];
            }

            sad += left_pixel.abs_diff(right_pixel) as u64;
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

    let padding = window_size / 2;
    let [left_buffer, right_buffer] = [left, right].map(|i| i.as_raw().clone());

    Ok((0..left.height() as i64)
        .into_par_iter()
        .map(|y| {
            calculate_disparities_across_y(
                &left_buffer,
                &right_buffer,
                padding,
                left.height() as i64,
                left.width() as i64,
                right.width() as i64,
                y,
            )
        })
        .flatten()
        .collect::<Vec<_>>())
}

fn scale_disparities(disparities: Vec<u64>) -> Result<Vec<u8>> {
    let max_value = disparities
        .iter()
        .max()
        .context("Disparity array is empty.")?;

    Ok(disparities
        .iter()
        .map(|i| ((i * u8::MAX as u64) / max_value) as u8)
        .collect::<Vec<u8>>())
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

fn save_disparity_map_as_image(
    scaled_disparities: Vec<u8>,
    width: u32,
    height: u32,
    filename: &str,
) -> Result<()> {
    let image: ImageBuffer<Luma<u8>, Vec<u8>> =
        ImageBuffer::from_vec(width, height, scaled_disparities)
            .context("Failed to make image buffer from disparity map.")?;
    Ok(image.save(filename)?)
}

fn main() -> Result<()> {
    let (left_image, right_image) = open_images("teddyL.pgm", "teddyR.pgm")?;
    let disparities = calculate_disparities(&left_image, &right_image, 3)?;
    let scaled_disparities = scale_disparities(disparities)?;
    save_disparity_map_as_image(
        scaled_disparities,
        left_image.width(),
        left_image.height(),
        "disparities.png",
    )?;

    Ok(())
}
