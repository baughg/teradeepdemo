--[[ frame

The 'frame' package has two public functions. The first is the 'init' function
that takes as arguments the application options table and a second source
definitions table. The second public function is a 'forward' function that is
created by the the 'init' function using the options/source tables to select an
appropriate method. It is this 'forward' function that is called by the
applications 'main' loop to grab an image frame.

--]]
local frame = {}
torch.setdefaulttensortype('torch.FloatTensor')

local sys = assert(require('sys'))

local pf = function(...) print(string.format(...)) end
local Cr = sys.COLORS.red
local Cb = sys.COLORS.blue
local Cg = sys.COLORS.green
local Cn = sys.COLORS.none
local THIS = sys.COLORS.blue .. 'THIS' .. Cn


local function prep_lua_camera(opt, source)
   assert(require('camera'))

   local cam = image.Camera {
      idx      = opt.cam,
      width    = source.w,
      height   = source.h,
   }

   -- set frame forward function
   frame.forward = function(img)
      return cam:forward(img)
   end

   source.cam = cam
end


local function prep_libcamera_tensor(opt, source)
   local cam = assert(require("libcamera_tensor"))

   if not cam.init(opt.cam, source.w, source.h, opt.fps, 1) then
      error("No camera")
   end

   -- set frame forward function
   frame.forward = function(img)
      if not cam.frame_rgb(opt.cam, img) then
         error('frame grab failed')
      end

      return img
   end

   source.cam = cam
end


local function prep_lua_linuxcamera(opt, source)
   local cam = assert(require('linuxcamera'))

   -- buffers default == 1
   cam.capture('/dev/video'..opt.cam, source.w, source.h, source.fps)

   -- set frame forward function
   frame.forward = function(img)
      img = torch.FloatTensor(3, source.h, source.w)
      cam.frame_rgb(img)
      return img
   end

   source.cam = cam
end


function frame:init(opt, source)
   if (opt.input == 'usbcam') and (sys.OS == 'macos') then

      prep_lua_camera(opt, source)
      
   elseif (opt.input == 'usbcam') then

      prep_lua_linuxcamera(opt, source)

   else
      error("<ERROR frame> Unsupported platform and frame source combination")
   end
end


return frame
