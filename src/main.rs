use anyhow::{ensure, Result};
use image::io::Reader as ImageReader;
use image::ImageBuffer;
use image::Luma;
use rayon::prelude::*;

#[inline]
fn abs_dif<T>(v1: T, v2: T) -> T
where
    T: PartialOrd + std::ops::Sub<Output = T> + Copy,
{
    if v1 > v2 {
        v1 - v2
    } else {
        v2 - v1
    }
}

#[inline]
fn calculate_sad(
    left: &ImageBuffer<Luma<u8>, Vec<u8>>,
    right: &ImageBuffer<Luma<u8>, Vec<u8>>,
    left_x: u32,
    right_x: u32,
    y: u32,
    padding: i64,
) -> usize {
    let mut sad: usize = 0;
    for offset_x in -padding..=padding {
        for offset_y in -padding..=padding {
            let final_left_x = left_x as i64 + offset_x;
            let final_right_x = right_x as i64 + offset_x;
            let final_y = y as i64 + offset_y;

            let left_pixel;
            let right_pixel;

            if final_left_x < 0 || final_left_x >= left.width() as i64 {
                left_pixel = 0;
            } else if final_y < 0 || final_y >= left.height() as i64 {
                left_pixel = 0;
            } else {
                left_pixel = left.get_pixel(final_left_x as u32, final_y as u32).0[0];
            }

            if final_right_x < 0 || final_right_x >= right.width() as i64 {
                right_pixel = 0;
            } else if final_y < 0 || final_y >= right.height() as i64 {
                right_pixel = 0;
            } else {
                right_pixel = right.get_pixel(final_right_x as u32, final_y as u32).0[0];
            }

            let ad = if left_pixel > right_pixel {
                left_pixel - right_pixel
            } else {
                right_pixel - left_pixel
            };

            sad += ad as usize;
        }
    }
    sad
}

fn main() -> Result<()> {
    // let img = ImageReader::open("disp2.pgm").unwrap().decode().unwrap();
    // img.save("disp2.jpg").unwrap();
    let left = ImageReader::open("teddyL.pgm")?.decode()?.into_luma8();
    let right = ImageReader::open("teddyR.pgm")?.decode()?.into_luma8();
    ensure!(
        left.height() == right.height(),
        "Image heights must be equal."
    );

    let padding: i64 = 1;

    let disparities = (0..left.height())
        .into_par_iter()
        .map(|y| {
            (0..left.width())
                .into_iter()
                .map(|left_x| {
                    (0..right.width())
                        .into_iter()
                        .fold((usize::MAX, 0), |(best_sad, best_disparity), right_x| {
                            let sad = calculate_sad(&left, &right, left_x, right_x, y, padding);
                            if sad < best_sad {
                                (sad, abs_dif(left_x, right_x))
                            } else {
                                (best_sad, best_disparity)
                            }
                        })
                        .1
                })
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect::<Vec<_>>();

    println!("{disparities:#?}");
    let max_value = disparities.iter().max().unwrap();
    let scaled_disparities = disparities
        .iter()
        .map(|i| ((i * 255) / max_value) as u8)
        .collect::<Vec<u8>>();
    let img_edited: ImageBuffer<Luma<u8>, Vec<u8>> =
        ImageBuffer::from_vec(left.width(), left.height(), scaled_disparities).unwrap();
    img_edited.save("disparities.png").unwrap();

    Ok(())
}

// let img = ImageReader::open("teddyL.pgm").unwrap().decode().unwrap();
// let mut array = img.as_bytes().to_vec();
// let width = img.width();
// let height = img.height();
// for i in 0..100 {
//     array[i] = 0;
// }
// let img_edited: ImageBuffer<Luma<u8>, Vec<u8>> =
//     ImageBuffer::from_vec(width, height, array).unwrap();
// img_edited.save("teddyL_edited.jpg").unwrap();
// img.save("teddyL.jpg").unwrap();
