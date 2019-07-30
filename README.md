# StitchIt!

## Background

This repository holds the code for a little STEAM (Science Technology Engineering Arts and Math) hobby project I started in the Spring of 2015 to combine my loves of coding, counted cross-stitch, and particle physics.  You know, like you do.  I really wanted to make a counted cross-stitch pattern from a particle collision event display image.  Sure, there are online and commercial software packages that I could have used to get the pattern without much work, but this was a case where I could easily envision how to write the code and I was really excited to write my own pattern creator.  No different from cooking fancy food at home in the kitchen.  Sure I could go to a restaurant and get a professional to make it for me, but where's the fun in that?  Well, actually that *is* a lot of fun, but so is cooking your own masterpieces.  As a bonus, the coding skills I honed on this personal project are directly applicable to my professional work, so there's that.

Anyway, the code you find here will create a counted cross-stitch pattern from an image by pixellating it, determining color maps and replacing each pixel by a symbol on a pattern grid for each color. One can choose how many different colors to allow for a given image so that it is possible to minimize the complexity. The pattern is output to a file called Pattern.pdf.

I provided a JuPyteR notebook [CrossStitch.ipynb](./CrossStitch.ipynb) with a step-by-step example of how the code is used to create the pattern, showing the steps of the process, with helpful display output. To create your own pattern, just modify the list of inputs and execute the remaining code cells. The default pattern created here is from a Higgs Boson Event image on wikipedia. If you want to skip the "how it works" part, the pattern can also be created with one call to the `testme` function provided with the code library.  Note that the pattern might be slightly different each time you run the code because the color reduction algorithm has some randomness associated with how it seeds the vector of colors.  In particular, the background is sometimes rendered as an RGB tuple other than [0, 0, 0], aka black.  Running the cell to reduce the colors a few times usually fixes that problem.

Speaking of which, some of the code is still rough.  For example, since it was written with the Higgs Boson event as the template, the black background color is treated specially.  I'd like to make it more robust so that it can handle any image file with any background color. It could also use some more documentation, and there are a few other quirks as well, but overall I am pretty happy with it.  If you have any ideas and want to contribute to making it better, let me know.

## Update July 2019
I migrated the code in the summer of 2019 to Python 3.7 because I finally finished stitching the masterpiece and entered the work in the 2019 California Mid-State Fair. I knew that the code was long overdue for this update so on a random Wednesday in July I decided to look at it again.  There are still some rough edges but at least it runs and produces a meaningful output.

The pattern that I stitched is available as a pdf below.  Some notes
about it:  I used 14-count Aida cloth in black, so I did not need to buy 21
skeins of black thread.  I bought the cloth online because the size I needed is
not available in typical craft stores like Michael's or Beverly's. I recommend
starting at the middle of the pattern (page 6). I also ignored the border line 
that appears on some edges of the original image.

[Pattern Link (pdf)](./img/HiggsPattern-AsStitched.pdf)

A colleague asked me why I chose this particular graphic, since it is a simulation, 
not an actual collision event display, fairly old (circa 2000), and not that accurate.
My reasons for choosing it were three-fold:  
1. it has a great color palette - better than many of the actual event displays 
2. it has been the main image used on CERN gift shop t-shirts for many years so it is well-known 
3. it is not tied to only one experiment but represents the modern LHC era honoring all high energy particle/nuclear physics. (There are lots of iconic bubble chamber images that I may attempt in the future because they are pretty.)

I will probably try doing pieces specific to my experiments at some point but they have to be the right ones. 
[ALICE Pb-Pb event displays](https://cds.cern.ch/record/2202730) are rather bland, color-wise, 
even if they are impressive.

## Image gallery

Here are some images of the completed stitching in the frame I found for it and used to display at the 
2019 California Mid-State Fair.  It won third place for needlework. Next stop: my office wall.

| ------------- |:-------------:| 
| ![Stitched Image 4][img4]      | ![Stitched Image 1][img1] | 
| The completed work before submission to the 2019 California Mid-State Fair. | Stitched pattern on display at the 2019 California Mid-State Fair. |
| ![Stitched Image 3][img3]       | ![Stitched Image 2][img2] |
| Me with my work at the 2019 California Mid-State Fair. | Stitched pattern on display at the 2019 California Mid-State Fair.  The Chart of Nuclides (upper right) was my other entry but it didn't place. |

[img1]: https://github.com/dedx/StitchIt/raw/master/img/image001.jpg "Stitched Image 1"
[img2]: https://github.com/dedx/StitchIt/raw/master/img/image002.jpg "Stitched Image 2"
[img3]: https://github.com/dedx/StitchIt/raw/master/img/image003.jpg "Stitched Image 3"
[img4]: https://github.com/dedx/StitchIt/raw/master/img/image004.jpg "Stitched Image 4"


Happy stitching!

J.L. Klay
29-July-2019

<hr>

© 2019 StitchIt - [https://github.com/dedx/StitchIt](https://github.com/dedx/StitchIt)