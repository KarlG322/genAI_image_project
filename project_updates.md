4/20: got the dataset I plan to work from (a few thousand patent figure images from 1 of the USPTO's APIs). They are pretty much all black and white line drawings because that's what the USPTO prefers. Not uploading to GitHub because file sizes are too large. A few notes on this:
- All figures are from technology center 3700 which handles a wide variety of mechanical devices, everything from shoes to helicopters. I figured this would make for more interesting figures than, for example, some of the pharma art units which would include more charts and experiment results and less diagrams of their device.
- I'm taking a maximum of 5 figures from any patent, so no patent is overly influential in the dataset.
- I'm only using figures from issued patents, not including patent applications.
- This isn't easy for others to reproduce, at least not the way I'm doing it. That's because it requires a USPTO ODP API key, which most people won't have.

4/27: 
- Manually removed bad data from the dataset It was originally 3000 figures but after removing things that aren't supposed to be there like the text of the body of the patent, 1 random other form, and a couple photos, it's 2948 figures. The type of document I pulled last week is really only supposed to be black and white line drawings.
- Converted PDFs of these figures to 128x128 grayscale PNGs.
- Made a basic diffusion model that should work on this dataset we'll see.
