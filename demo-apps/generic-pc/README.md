# demonstration application

This is our January 2016 top neural network for large-scale object recognition. It has been trained to recognize most typical home indoor/outdoor objects in our daily life. It was trained with more that 32 M images on a private dataset. It can serve as good pair of eyes for your machines, robots, drones and all your wonderful creations!

<!--[![icon](icon.jpg|width=400px)]-->
<a href="icon"><img src="icon.jpg" align="center" height="400" width="400" ></a>

See it in action in this [video #1](https://www.youtube.com/watch?v=_wXHR-lad-Q), and also this other [video #2](https://www.youtube.com/watch?v=B0TreumQO-0).

This application is for tinkerers, hobbiest, researchers, evaluation purpose, non-commercial use only.

It has been tested on OS X 10.10.3 and Linux. It can run at > 17 fps on a MacBook Pro (Retina, 15-inch, Late 2013) on CPU only.


## install:
Install Torch7: http://torch.ch/

Please download files: `model.net`, `categories.txt` and `stat.t7` from: https://www.dropbox.com/sh/u3bunkfm0dzjix6/AABQ4Nq4-70MU57MxXyyjrrMa?dl=0
This is our new 5000 categories network from January 2016.

Also this is our older 1000 categories model:
https://www.dropbox.com/sh/qw2o1nwin5f1r1n/AADYWtqc18G035ZhuOwr4u5Ea?dl=0

Linux camera install: `cd lib/` then `make; make install`. Note that `Makefile` wants Torch7 installed in `/usr/local/bin`, otherwise please change accordingly!


## run:
To run with a webcam and display on local machine: ```qlua run.lua```

Zoom window by 2 (or any number): ```qlua run.lua -z 2```


## usage:

Feel free to modify and use for all you non-commercial projects. Interested parties can license this and other Teradeep technologies by contacting us at `info@teradeep.com`

## most importantly:

Have fun! Life is short, we need to produce while we can!

## credits:
Aysegul Dundar, Jonghoon Jin, Alfredo Canziani, Eugenio Culurciello, Berin Martini all contributed to this work and demonstration. Thank you all!
