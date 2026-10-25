# AI Image Changer - Evaluation Harness

This directory contains the testing harness, benchmark dataset, and regression evaluation suite for background removal and cutout refinement.

## Directory Structure
- `images/`: Evaluation dataset partitioned into 15 domain categories:
  - `portrait_closeup/`: Single face/bust portraits with varied skin tones, glasses, expressions.
  - `full_body/`: Standing, sitting, walking people with varying poses and limbs.
  - `group/`: Multi-person images (couples, teams, family photos).
  - `hair_fine/`: Flyaway hair, curls, afro, thin wisps, backlighting.
  - `product_clean/`: E-commerce product photos on solid or smooth studio backgrounds.
  - `product_busy/`: Products on complex, textured, patterned, or shelf backgrounds.
  - `pet_fur/`: Dogs, cats, horses with fine fur boundaries and whiskers.
  - `food/`: Dishes, beverages, fruits, transparent garnishes, steam/gloss.
  - `vehicle/`: Cars, bikes, scooters with metallic reflections and ground shadows.
  - `poster_graphic_text/`: Collages, text banners, standees (includes the `rang_morla.jpg` regression sample).
  - `low_light/`: Dimly lit indoor/night photos with high sensor noise and low edge contrast.
  - `transparent_glass/`: Bottles, wine glasses, eyewear, semi-transparent plastics.
  - `low_contrast/`: Foreground subject color very similar to background.
  - `small_subject/`: Small subject occupying <10% of image area.
  - `large_4k_photo/`: High-resolution photos (>12MP / 4K) to measure memory scaling and speed.

## Adding Test Images
- Drop 8–15 representative JPG or PNG images into each category folder.
- **Ground-Truth Masks (Optional)**: If you have an exact ground truth mask, name it `<image_stem>.mask.png` in the same directory (grayscale single-channel or RGBA where 255 = foreground, 0 = background).
- If no ground truth mask is provided, the evaluation harness calculates:
  - **Artifact score**: Ratio of uncertain alpha band pixels (0.1–0.9).
  - **Component fragmentation**: Count of isolated foreground islands.
  - **Edge-halo measure**: Mean color bleed distance between foreground edge band and removed background.

## Running the Evaluation Harness
Run from the `backend/` directory:
```powershell
python -m eval.run_eval --pipeline legacy
python -m eval.run_eval --pipeline balanced
python -m eval.run_eval --pipeline all
```

Output contact sheets and results will be saved in:
`backend/eval/results/<pipeline_name>/`
Each test image produces a 5-panel contact sheet:
`[Original | Alpha Mask | Checkerboard | Magenta (#FF00FF) | Dark Green (#003300)]`
