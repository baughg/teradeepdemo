--------------------------------------------------------------------------------
-- Demo for generic teradeep detector
-- E. Culurciello, May 2015
-- with help from A. Canziani, JH Jin, A. Dundar, B. Martini
--------------------------------------------------------------------------------

-- Requires --------------------------------------------------------------------
require 'pl'
require 'nn'
require 'sys'
require 'paths'
require 'image'
local frame = assert(require('frame'))
local display = assert(require('display'))
local process = assert(require('process'))

-- Local definitions -----------------------------------------------------------
local pf = function(...) print(string.format(...)) end
local Cr = sys.COLORS.red
local Cb = sys.COLORS.blue
local Cg = sys.COLORS.green
local Cn = sys.COLORS.none
local THIS = sys.COLORS.blue .. 'THIS' .. Cn

-- Title definition -----------------------------------------------------------
title = [[
 _____                  _
|_   _|                | |
  | | ___ _ __ __ _  __| | ___  ___ _ __
  | |/ _ \ '__/ _` |/ _` |/ _ \/ _ \ '_ \
  | |  __/ | | (_| | (_| |  __/  __/ |_) |
  \_/\___|_|  \__,_|\__,_|\___|\___| .__/
                                   | |
                                   |_|
]]

-- Options ---------------------------------------------------------------------
opt = lapp(title .. [[
--mode           (default 'pc')               Use mode: pc, device, server
--nt             (default 8)                  Number of threads for multiprocessing
--is             (default 231)                Image's side length
--camRes         (default QHD)                Camera resolution (QHD|VGA|FWVGA|HD|FHD)
-i, --input      (default usbcam)             Input is USB camera (usbcam), or video (video)
-c, --cam        (default 0)                  Camera device number
-z, --zoom       (default 1)                  Zoom ouput window
--pdt            (default 0.5)                Detection threshold to detect person vs background
--fps            (default 30)                 Frames per second (camera setting)
--gui            (default true)               Use GUI display (default false) which sends to server
--dv             (default 5)                  Verbosity of detection: 0=1target, >0:#top categories to show
--diw            (default 6)                  number of frames for integration of object detection
]])

pf(Cb..title..Cn)
torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(opt.nt)
print('Number of threads used:', torch.getnumthreads())


-- global objects:
local network = {} -- network object
source = {} -- source object

-- Loading neural network:
network.model = torch.load('model.net')
-- Loading classes names and also statistics from dataset
network.stat = torch.load('stat.t7')

-- change targets based on categories csv file:
function readCatCSV(filepath)
   local file = io.open(filepath,'r')
   local classes = {}
   local targets = {}
   file:read() -- throw away first line of category file
   local fline = file:read()
   while fline ~= nil do
      local col1, col2 = fline:match("([^,]+),([^,]+)")
      table.insert(classes, col1)
      table.insert(targets, ('1' == col2))
      fline = file:read()
   end
   return classes, targets
end

-- load categories from file folder or this (run.lua) folder
network.classes, network.targets = readCatCSV('categories.txt')

-- switch input sources
source.res = {
   HVGA  = {w =  320, h =  240},
   QHD   = {w =  640, h =  360},
   VGA   = {w =  640, h =  480},
   FWVGA = {w =  854, h =  480},
   HD    = {w = 1280, h =  720},
   FHD   = {w = 1920, h = 1080},
}
source.w = source.res[opt.camRes].w
source.h = source.res[opt.camRes].h
source.fps = opt.fps
local src = torch.FloatTensor(3, source.h, source.w)

-- init application packages
frame:init(opt, source)
display:init(opt, source, network.classes, network.targets)
process:init(opt, source, network)

-- profiling timers
local timer = torch.Timer()
local t_loop = 1 -- init to 1s

-- create main functions
local main = function()
   status, err = pcall(function()
      while display.continue() do
         timer:reset()

         src = frame.forward(src)
         if not src then
            break
         end

         local result, img = process.forward(src)
         
         if img:dim() == 3 then
            display.forward(result, img, (1/t_loop))
         else
            for i = 1,result:size(1) do
               display.forward(result[i], img[i], (result:size(1)/t_loop))
            end
         end

         t_loop = timer:time().real

         collectgarbage()
      end
   end)
   if status then
      print('process done!')
   else
      print('Error ' .. err)
   end
   display.close()
end

-- execute main loop
main()
