use image::io::Reader as ImageReader;
use image::ImageBuffer;
use image::Luma;

fn main() {
    // let img = ImageReader::open("disp2.pgm").unwrap().decode().unwrap();
    // img.save("disp2.jpg").unwrap();
    let left = ImageReader::open("teddyL.pgm")
        .unwrap()
        .decode()
        .unwrap()
        .into_luma8();
    let right = ImageReader::open("teddyR.pgm")
        .unwrap()
        .decode()
        .unwrap()
        .into_luma8();

    let mut correspondences = vec![];
    let mut disparities = vec![];
    println!("{disparities:#?}");

    let padding: i64 = 1;
    for y in 0..left.height() {
        println!("{y}");
        for left_x in 0..left.width() {
            let mut best_sad = u32::MAX;
            let mut best_x = 0;
            for right_x in 0..right.width() {
                let mut sad: u32 = 0;
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
                            right_pixel =
                                right.get_pixel(final_right_x as u32, final_y as u32).0[0];
                        }

                        let ad = if left_pixel > right_pixel {
                            left_pixel - right_pixel
                        } else {
                            right_pixel - left_pixel
                        };

                        sad += ad as u32;
                    }
                }
                if sad < best_sad {
                    best_sad = sad;
                    best_x = right_x;
                }
            }
            correspondences.push(best_x);
            disparities.push(if best_x > left_x {
                best_x - left_x
            } else {
                left_x - best_x
            });
        }
    }

    println!("{disparities:#?}");
    let max_value = disparities.iter().max().unwrap();
    let scaled_disparities = disparities
        .iter()
        .map(|i| ((i * 255) / max_value) as u8)
        .collect::<Vec<u8>>();
    let img_edited: ImageBuffer<Luma<u8>, Vec<u8>> =
        ImageBuffer::from_vec(left.width(), left.height(), scaled_disparities).unwrap();
    img_edited.save("disparities.png").unwrap();
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
